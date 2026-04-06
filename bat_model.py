#Battery schedule model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data and convert from 15 minute to hourly data

#load excel files from same folder as python script
base_dir = os.path.dirname(os.path.abspath(__file__))

generation_file = os.path.join(base_dir, "renewable_generation_data2.xlsx")
demand_file = os.path.join(base_dir, "factory_demand.xlsx")

#load renewable generation excel file
gen_df = pd.read_excel(generation_file)

print("Generation columns:", gen_df.columns.tolist())

#Date_Time is already in the file, so just convert it directly
gen_df["Date_Time"] = pd.to_datetime(gen_df["Date_Time"], errors="coerce")

#drop rows where datetime could not be read
gen_df = gen_df.dropna(subset=["Date_Time"])

#Set datetime as index
gen_df = gen_df.set_index("Date_Time").sort_index()

#keep only renewable generation columns
gen_df = gen_df[["Wind_generation_KW", "Solar_generation_KW"]]


#convert 15 minute data to hourly data
#because the data is power in kW, take hourly mean power
gen_hourly = gen_df.resample("h").mean()

#make columns easier to use in model
gen_hourly["datetime"] = gen_hourly.index

#convert generation from kW to MW
gen_hourly["wind_mw"] = gen_hourly["Wind_generation_KW"] / 1000
gen_hourly["solar_mw"] = gen_hourly["Solar_generation_KW"] / 1000


#load factory demand excel file
demand_df = pd.read_excel(demand_file, header=1)

print("Demand columns before rename:", demand_df.columns.tolist())

#rename columns to simpler names
demand_df.columns = ["day", "time_h", "electricity_mw"]

#clean columns
demand_df["day"] = demand_df["day"].astype(str).str.strip().str.capitalize()

#convert time column
demand_df["time_h"] = pd.to_datetime(
    demand_df["time_h"].astype(str),
    format="%H:%M:%S",
    errors="coerce"
).dt.time

#ensure demand values are numeric
demand_df["electricity_mw"] = pd.to_numeric(
    demand_df["electricity_mw"], errors="coerce"
)

#drop bad rows
demand_df = demand_df.dropna(subset=["day", "time_h", "electricity_mw"])


#align factory demand profile with renewable generation timestamps
gen_hourly["day"] = gen_hourly["datetime"].dt.day_name()
gen_hourly["time_h"] = gen_hourly["datetime"].dt.time

model_df = pd.merge(
    gen_hourly.reset_index(drop=True),
    demand_df[["day", "time_h", "electricity_mw"]],
    on=["day", "time_h"],
    how="left"
)

model_df = model_df.rename(columns={"electricity_mw": "demand_mw"})

#check if any demand values did not match
missing_demand = model_df["demand_mw"].isna().sum()
print(f"Missing demand rows after merge: {missing_demand}")

#fill missing demand values using nearest available values
model_df["demand_mw"] = model_df["demand_mw"].ffill().bfill()

print("Model columns:", model_df.columns.tolist())
print(model_df.head())

#extract numpy arrays for model
wind = model_df["wind_mw"].to_numpy()
solar = model_df["solar_mw"].to_numpy()
demand = model_df["demand_mw"].to_numpy()


#Weather forecasting functions
#wind forecast average wind generation for last 24 hours

def forecast_renewables(wind, solar, t, horizon=24, lookback=24):

    start_idx = max(0, t - lookback)  # so we dont go into negative timestep

    #wind forecast
    wind_forecast = np.mean(wind[start_idx:t]) if t > start_idx else 0

    forecast = []

    for h in range(1, horizon + 1):

        future_idx = t + h

        #Solar forecast- use yesterdays same hour value as proxy for future solar generation
        if future_idx - 24 >= 0 and future_idx - 24 < len(solar):
            solar_forecast = solar[future_idx - 24]
        else:
            solar_forecast = 0

        #total renewable forecast for each future hour
        forecast.append(wind_forecast + solar_forecast)

    return np.array(forecast)


#forecast factory demand using same hour value from previous day as proxy

def forecast_demand(demand, t, horizon=24):

    forecast = []

    for h in range(1, horizon + 1):
        future_idx = t + h

        if future_idx - 24 >= 0 and future_idx - 24 < len(demand):
            demand_forecast = demand[future_idx - 24]
        else:
            demand_forecast = demand[t]

        forecast.append(demand_forecast)

    return np.array(forecast)


#next: need to determine how much battery charge to keep in the reserve to use over next 24 hours
#soc = state of charge at battery at time t (as %)
#reserve soc = percentage of battery capacity to keep in reserve for future use

def calc_reserve_soc_from_excess(forecast_excess, battery_capacity, high_excess_threshold, low_excess_threshold):

    if forecast_excess >= high_excess_threshold:
        reserve_soc = 0.10 * battery_capacity  # keep only 10% battery in reserve if we have a lot of excess

    elif forecast_excess <= low_excess_threshold:
        reserve_soc = 0.80 * battery_capacity  # keep 80% battery in reserve if we have very little excess / shortage expected

    else:
        reserve_soc = 0.40 * battery_capacity  # keep 40% battery in reserve if we have moderate excess

    return reserve_soc


#Battery dispatch function to determine whether to discharge battery at current time t
#battery capacity- max battery capacity in MWh
#max charge power - max power battery can charge in one hour in MW
#max discharge power- max power battery can discharge in one hour in MW
#charge efficiency - percentage power lost to heat when charging battery 0.95 (5%) loss
#discharge efficiency- percentage power lost to heat when discharging battery 0.95
#export limit- max power battery can export to grid in one hour in MW - 5MW
#initial SOC - initial state of charge of battery at time t in MWh
#demand_t demand at time t in MW

def battery_dispatch(
    wind,
    solar,
    demand,
    battery_capacity,
    max_charge_power,
    max_discharge_power,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    export_limit=5,
    initial_soc=0.0,
    high_excess_threshold=5,
    low_excess_threshold=2,
    horizon=24
):

    n = len(demand)
    soc = initial_soc
    results = []

    for t in range(n):

        #current hour renewable generation and demand
        renewable_generation = wind[t] + solar[t]
        demand_t = demand[t]
        excess = renewable_generation - demand_t

        #forecast future renewable generation and demand
        forecasted_generation = forecast_renewables(wind, solar, t, horizon=horizon)
        forecasted_demand = forecast_demand(demand, t, horizon=horizon)

        #calculate forecasted excess for next 24 hours
        #positive excess = more renewable generation than demand
        #negative excess = more demand than renewable generation
        forecasted_excess = forecasted_generation.sum() - forecasted_demand.sum()

        #determine how much battery charge to keep in reserve based on forecasted excess
        reserve_soc = calc_reserve_soc_from_excess(
            forecast_excess=forecasted_excess,
            battery_capacity=battery_capacity,
            high_excess_threshold=high_excess_threshold,
            low_excess_threshold=low_excess_threshold
        )

        #keep soc within physical battery limits
        soc = max(0, min(soc, battery_capacity))

        #initialise hourly outputs
        charge = 0.0
        discharge = 0.0
        export = 0.0
        curtailment = 0.0
        grid_import = 0.0

        #IF SURPLUS EXCESS RENEWABLE GENERATION
        if excess > 0:

            #battery can only accept enough input to fill remaining space after efficiency losses
            available_charge_input = (battery_capacity - soc) / charge_efficiency

            charge = min(excess, max_charge_power, available_charge_input)
            soc += charge * charge_efficiency

            #snap to full if extremely close
            if abs(soc - battery_capacity) < 1e-6:
                soc = battery_capacity

            #export remaining excess to grid
            remaining_excess = excess - charge
            export = min(remaining_excess, export_limit)

            #curtail any excess beyond export limit
            curtailment = max(0, remaining_excess - export)

        #IF RENEWABLE GENERATION DEFICIT
        elif excess < 0:

            deficit = abs(excess)

            #discharge only above reserve level
            available_discharge = max(0, soc - reserve_soc)

            discharge = min(deficit, max_discharge_power, available_discharge)
            soc -= discharge / discharge_efficiency

            #import remaining deficit from grid
            grid_import = deficit - discharge

        #IF EXACT BALANCE BETWEEN GENERATION AND DEMAND
        else:
            pass

        #final soc clamp
        soc = max(0, min(soc, battery_capacity))

        results.append({
            "time": t,
            "wind_mw": wind[t],
            "solar_mw": solar[t],
            "demand_mw": demand[t],
            "renewable_generation_mw": renewable_generation,
            "excess_mw": excess,
            "forecasted_generation_sum": forecasted_generation.sum(),
            "forecasted_demand_sum": forecasted_demand.sum(),
            "forecasted_excess": forecasted_excess,
            "reserve_soc_mwh": reserve_soc,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "export_mw": export,
            "curtailment_mw": curtailment,
            "grid_import_mw": grid_import,
            "soc_end_mwh": soc
        })

    return pd.DataFrame(results)


#run battery dispatch model
results_df = battery_dispatch(
    wind=wind,
    solar=solar,
    demand=demand,
    battery_capacity=5,        #example battery size in MWh
    max_charge_power=5,        #example max charge rate in MW
    max_discharge_power=5,     #example max discharge rate in MW
    initial_soc=0.0            #example starting SOC in MWh
)

#add timestamps back to results
results_df["datetime"] = model_df["datetime"].values

#label each hour as excess / deficit / balanced
results_df["status"] = np.where(
    results_df["excess_mw"] > 0,
    "excess",
    np.where(results_df["excess_mw"] < 0, "deficit", "balanced")
)

#reorder columns for cleaner output
results_df = results_df[
    [
        "datetime",
        "wind_mw",
        "solar_mw",
        "demand_mw",
        "renewable_generation_mw",
        "excess_mw",
        "status",
        "forecasted_generation_sum",
        "forecasted_demand_sum",
        "forecasted_excess",
        "reserve_soc_mwh",
        "charge_mw",
        "discharge_mw",
        "export_mw",
        "curtailment_mw",
        "grid_import_mw",
        "soc_end_mwh"
    ]
]

#print first rows of output
print("\nFIRST 10 ROWS OF OUTPUT:\n")
print(results_df.head(10))

#save model outputs
csv_path = os.path.join(base_dir, "battery_dispatch_results2.csv")
xlsx_path = os.path.join(base_dir, "battery_dispatch_results2.xlsx")

results_df.to_csv(csv_path, index=False)

try:
    results_df.to_excel(xlsx_path, index=False)
    print(f"Excel saved to: {xlsx_path}")
except PermissionError:
    print(f"Could not save Excel file because it is open: {xlsx_path}")
    print("Close the Excel file and run again, or save with a different filename.")

print("\nModel run complete.")
print("Saved files:")
print(csv_path)
print(xlsx_path)


#plot results

#plot results

#make sure datetime is datetime format
results_df["datetime"] = pd.to_datetime(results_df["datetime"])

# PLOT 1: battery SOC
plt.figure(figsize=(12, 5))
plt.plot(results_df["datetime"], results_df["soc_end_mwh"])
plt.title("Battery State of Charge Over Time")
plt.xlabel("Datetime")
plt.ylabel("SOC (MWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PLOT 2: renewable generation vs demand
plt.figure(figsize=(12, 5))
plt.plot(results_df["datetime"], results_df["renewable_generation_mw"], label="Renewable generation (MW)")
plt.plot(results_df["datetime"], results_df["demand_mw"], label="Demand (MW)")
plt.title("Renewable Generation vs Demand")
plt.xlabel("Datetime")
plt.ylabel("Power (MW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PLOT 3: battery charge and discharge
plt.figure(figsize=(12, 5))
plt.plot(results_df["datetime"], results_df["charge_mw"], label="Charge (MW)")
plt.plot(results_df["datetime"], results_df["discharge_mw"], label="Discharge (MW)")
plt.title("Battery Charge and Discharge")
plt.xlabel("Datetime")
plt.ylabel("Power (MW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PLOT 4: export, curtailment, and grid import
plt.figure(figsize=(12, 5))
plt.plot(results_df["datetime"], results_df["export_mw"], label="Export to grid (MW)")
plt.plot(results_df["datetime"], results_df["curtailment_mw"], label="Curtailment (MW)")
plt.plot(results_df["datetime"], results_df["grid_import_mw"], label="Grid import (MW)")
plt.title("Grid Interaction")
plt.xlabel("Datetime")
plt.ylabel("Power (MW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
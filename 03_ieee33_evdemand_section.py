import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.lines as mlines


# -----------------------------
# IEEE-33 Load Data (kW, kVAr)
# -----------------------------
LOADS_IEEE33 = [
    (2, 100, 60),
    (3, 90, 40),
    (4, 120, 80),
    (5, 60, 30),
    (6, 60, 20),
    (7, 200, 100),
    (8, 200, 100),
    (9, 60, 20),
    (10, 60, 20),
    (11, 45, 30),
    (12, 60, 35),
    (13, 60, 35),
    (14, 120, 80),
    (15, 60, 10),
    (16, 60, 20),
    (17, 60, 20),
    (18, 90, 40),
    (19, 90, 40),
    (20, 90, 40),
    (21, 90, 40),
    (22, 90, 40),
    (23, 90, 50),
    (24, 420, 200),
    (25, 420, 200),
    (26, 60, 25),
    (27, 60, 25),
    (28, 60, 20),
    (29, 120, 70),
    (30, 200, 60),
    (31, 150, 70),
    (32, 210, 100),
    (33, 60, 40),
]


# -----------------------------
# Section definition (your split)
# -----------------------------
SECTIONS = {
    "Section 1": [2, 3, 4, 5, 19, 20, 21, 22, 23, 24, 25],
    "Section 2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "Section 3": [26, 27, 28, 29, 30, 31, 32, 33],
}


# -----------------------------
# EV cluster data (from your clustering)
# -----------------------------
CLUSTER_PERCENTAGES = np.array([0.3859, 0.2164, 0.3977])  # cluster share
CLUSTER_CAPACITIES_KWH = np.array([14.30, 66.06, 80.70])
CLUSTER_RANGES_KM = np.array([78, 236, 480])


# ============================================================
# 1) EV counts per bus & per section (penetration-sensitive)
# ============================================================
def estimate_evs_per_bus(loads_kw, penetration=0.3176, vehicles_per_dwelling=1.6):
    """
    This adapts your EV number logic to be penetration-sensitive.
    Your original code implicitly assumed a fixed multipliers; here we keep the same
    structure BUT scale the EV totals by penetration.
    """
    residential_percentage = 0.85
    commercial_percentage = 0.15
    avg_residential_consumption = 5
    avg_commercial_consumption = 20
    residential_ev_multiplier = 1.6
    commercial_ev_multiplier = 3.0
    safety_factor = 1.10

    households_and_evs = []
    for bus, load_kw, q_kvar in loads_kw:
        residential_load = load_kw * residential_percentage
        commercial_load = load_kw * commercial_percentage

        households = residential_load / avg_residential_consumption
        businesses = commercial_load / avg_commercial_consumption

        evs_residential = households * residential_ev_multiplier * penetration
        evs_commercial = businesses * commercial_ev_multiplier * penetration

        total_evs = (evs_residential + evs_commercial) * safety_factor
        households_and_evs.append((bus, households, businesses, evs_residential, evs_commercial, total_evs))

    return households_and_evs


def aggregate_evs_by_section(households_and_evs, sections):
    evs_per_section = {name: 0.0 for name in sections}
    evs_per_bus = {}

    for bus, households, businesses, evs_res, evs_com, total_evs in households_and_evs:
        evs_per_bus[bus] = total_evs
        for sec_name, buses in sections.items():
            if bus in buses:
                evs_per_section[sec_name] += total_evs

    # round for reporting
    evs_per_section_rounded = {k: int(round(v)) for k, v in evs_per_section.items()}
    evs_per_bus_rounded = {k: int(round(v)) for k, v in evs_per_bus.items()}

    return evs_per_section_rounded, evs_per_bus_rounded


# ============================================================
# 2) Uncertainty modelling (arrival time, daily distance, SOC)
# ============================================================
def plot_arrival_time_distribution(mean_arrival=15.65, std_arrival=5.39):
    samples_per_hour = 1000
    total_samples = samples_per_hour * 24
    arrival_times = np.random.normal(loc=mean_arrival, scale=std_arrival, size=total_samples).flatten()

    plt.figure(figsize=(6, 3))
    plt.hist(arrival_times, bins=30, color="cyan", alpha=0.5, edgecolor="black",
             density=True, label="Histogram")

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_arrival, std_arrival)
    plt.plot(x, p, "purple", label=f"Mean={mean_arrival}, Std={std_arrival}", linewidth=2.5)

    plt.xlim(0, 24)
    plt.xlabel("Arrival Time (Hour)", fontweight="bold")
    plt.ylabel("Probability Density", fontweight="bold")
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    plt.show()


def plot_daily_driven_distance(mu_ev=3.8959367126828115, sigma_ev=0.548785984602523):
    np.random.seed(42)
    data_daily_ev = np.random.lognormal(mu_ev, sigma_ev, 1000)

    plt.figure(figsize=(6, 4))
    counts_ev, bins_ev, _ = plt.hist(
        data_daily_ev, bins=30, density=True, alpha=0.6,
        color="purple", edgecolor="black", label="EV Daily Mileage"
    )

    x_ev = np.linspace(bins_ev.min(), bins_ev.max(), 1000)
    pdf_ev = lognorm.pdf(x_ev, sigma_ev, scale=np.exp(mu_ev))
    plt.plot(x_ev, pdf_ev, color="red", linestyle="-", linewidth=2.5, label="Fitted Lognormal PDF")

    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", prop={"family": "Arial", "weight": "bold"})
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.xlim(0, 200)
    plt.show()


def plot_initial_soc_distribution(mean_soc=45.90, std_soc=12.80):
    mu = np.log(mean_soc**2 / np.sqrt(std_soc**2 + mean_soc**2))
    sigma = np.sqrt(np.log(1 + (std_soc**2 / mean_soc**2)))

    initial_soc = np.random.lognormal(mu, sigma, 1000)

    plt.figure(figsize=(6, 4))
    plt.hist(initial_soc, bins=30, density=True, color="cyan", edgecolor="black",
             alpha=0.7, label="Histogram")

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 90)
    pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
    plt.plot(x, pdf, "r-", linewidth=2.5, label=f"Lognormal PDF\nMean={mean_soc}, Std={std_soc}")

    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.xlabel("Percentage of initial SOC", family="Arial", fontsize=12, fontweight="bold")
    plt.ylabel("Probability density", family="Arial", fontsize=12, fontweight="bold")
    plt.legend(prop={"family": "Arial", "weight": "bold"})
    plt.grid(True)
    plt.show()


# ============================================================
# 3) Plugin interval + daily arrivals per section (Monte Carlo)
# ============================================================
def simulate_section_plugin_intervals(
    total_evs,
    cluster_percentages=CLUSTER_PERCENTAGES,
    capacities=CLUSTER_CAPACITIES_KWH,
    ranges=CLUSTER_RANGES_KM,
    final_soc=90,
    initial_soc_mean=45.12,
    initial_soc_std=12.90,
    mean_driven=3.54,
    std_driven=0.65,
    num_simulations=1000,
):
    """
    Exact logic matching your "simulate_section" approach.
    Returns:
      avg_plugin_intervals_days (per cluster)
      avg_daily_arrivals_evs_per_day (per cluster)
      total_avg_daily_arrival_evs_per_day
    """
    num_ev_per_cluster = [total_evs * p for p in cluster_percentages]
    results = {f"Cluster {i+1}": [] for i in range(len(capacities))}

    for i in range(len(capacities)):
        battery_capacity = capacities[i]
        driving_range = ranges[i]
        num_ev = int(num_ev_per_cluster[i])
        energy_per_km = battery_capacity / driving_range

        for _ in range(num_simulations):
            initial_soc = np.clip(
                np.random.normal(loc=initial_soc_mean, scale=initial_soc_std, size=num_ev),
                0, 70
            )
            energy_needed = ((final_soc - initial_soc) / 100) * battery_capacity
            plugin_intervals = np.zeros(num_ev)

            for ev in range(num_ev):
                total_energy_consumed, days = 0, 0
                while total_energy_consumed < energy_needed[ev]:
                    daily_distance = min(
                        np.random.lognormal(mean=mean_driven, sigma=std_driven),
                        driving_range
                    )
                    total_energy_consumed += daily_distance * energy_per_km
                    days += 1
                plugin_intervals[ev] = days

            results[f"Cluster {i+1}"].append(np.mean(plugin_intervals))

    avg_plugin_intervals = {k: float(np.mean(v)) for k, v in results.items()}
    avg_daily_arrivals = [
        num_ev_per_cluster[i] / avg_plugin_intervals[f"Cluster {i+1}"] for i in range(len(capacities))
    ]
    total_avg_daily_arrival = float(np.sum(avg_daily_arrivals))

    return avg_plugin_intervals, avg_daily_arrivals, total_avg_daily_arrival


# ============================================================
# 4) EV demand profiles per section
#    (A) hourly (24)
#    (B) half-hourly (48)
# ============================================================
def hourly_ev_demand_profile(
    total_evs,
    penetration=0.3176,
    capacities=CLUSTER_CAPACITIES_KWH,
    cluster_percentages=CLUSTER_PERCENTAGES,
    power_levels=np.array([3.6, 7.4, 11]),
    mean_arrival_time=16.55,
    std_arrival_time=5.79,
    mean_initial_soc=0.45,
    std_initial_soc=0.1290,
    final_soc=0.9,
    charging_levels_prob=np.array([0.41, 0.54, 0.05]),
    charger_efficiency=0.95,
    num_iterations=1000,
):
    """
    Your hourly EV demand code adapted so total_evs comes from IEEE-33 section estimation.
    Returns mean profile across Monte Carlo iterations (kW per hour).
    """
    num_evs = int(total_evs * penetration)

    hourly_load_demand = np.zeros((num_iterations, len(capacities), 24))

    for sim in range(num_iterations):
        arrival_times = np.random.normal(mean_arrival_time, std_arrival_time, num_evs)
        arrival_times = np.mod(arrival_times, 24)

        initial_soc = np.clip(np.random.normal(mean_initial_soc, std_initial_soc, num_evs), 0, 0.7)

        start = 0
        for cluster_idx, capacity in enumerate(capacities):
            cluster_evs = int(num_evs * cluster_percentages[cluster_idx])
            cluster_arrivals = arrival_times[start:start + cluster_evs]
            cluster_initial_soc = initial_soc[start:start + cluster_evs]
            start += cluster_evs

            profile = np.zeros(24)

            for hour in range(24):
                evs_arrived_idx = np.where(np.floor(cluster_arrivals) == hour)[0]
                num_arrivals = len(evs_arrived_idx)
                if num_arrivals == 0:
                    continue

                available_charging_time = (hour + 1) - cluster_arrivals[evs_arrived_idx]
                available_charging_time = np.clip(available_charging_time, 0, 1)

                soc = cluster_initial_soc[evs_arrived_idx]

                energy_needed = capacity * (final_soc - soc)
                energy_needed = np.clip(energy_needed, 0, None)

                charging_powers = np.random.choice(power_levels, size=num_arrivals, p=charging_levels_prob)

                max_energy_possible = charging_powers * available_charging_time * charger_efficiency
                energy_delivered = np.minimum(energy_needed, max_energy_possible)

                power_draw = np.where(
                    available_charging_time > 0,
                    energy_delivered / available_charging_time,
                    0
                )

                profile[hour] += np.sum(power_draw)

            hourly_load_demand[sim, cluster_idx, :] = profile

    mean_hourly = np.mean(hourly_load_demand, axis=0)  # [cluster, hour]
    mean_total = np.sum(mean_hourly, axis=0)          # [hour]
    return mean_total, mean_hourly


def half_hourly_section_demand(
    total_evs,
    penetration=0.3792,
    capacities=(14.30, 66.06, 80.70),
    percentages=(0.3859, 0.2164, 0.3977),
    power_levels=(22, 50),
    mean_arrival_time=16.55,
    std_arrival_time=5.79,
    mean_initial_soc=0.4520,
    std_initial_soc=0.1290,
    final_soc=0.9,
    charger_efficiency=0.95,
    num_iterations=100,
):
    """
    Matches your half-hourly demand simulation for each section, output length 48.
    Returns mean_total_half_hourly_demand (kW per 30-min interval).
    """
    num_evs = int(total_evs * penetration)

    total_half_hourly_demand = np.zeros((num_iterations, 48))

    for sim in range(num_iterations):
        arrival_times = np.random.normal(loc=mean_arrival_time, scale=std_arrival_time, size=num_evs)
        arrival_times = np.mod(arrival_times, 24)

        for cluster in range(len(capacities)):
            charging_rate = max(power_levels)

            for half_hour in range(48):
                hour = half_hour // 2
                num_arrivals = np.sum((arrival_times >= hour) & (arrival_times < hour + 0.5))

                num_cluster_arrivals = int(num_arrivals * percentages[cluster])

                for _ in range(num_cluster_arrivals):
                    initial_soc = np.random.normal(loc=mean_initial_soc, scale=std_initial_soc)
                    charging_time = (final_soc - initial_soc) / charging_rate

                    start_half_hour = half_hour
                    end_half_hour = min(48, start_half_hour + int(np.ceil(charging_time * 2)))

                    for h in range(start_half_hour, end_half_hour):
                        if h < 48:
                            actual_charging_power = charging_rate * charger_efficiency
                            total_half_hourly_demand[sim, h] += actual_charging_power

    mean_total = np.mean(total_half_hourly_demand, axis=0)
    return mean_total


# ============================================================
# 5) Energy demand probability (kWh) Monte Carlo
# ============================================================
def energy_demand_probability(
    iterations=1000,
    battery_capacities=(14.30, 66.06, 80.70),
    cluster_probs=(0.3859, 0.2164, 0.3977),
    mean_initial_soc=0.4590,
    std_dev_initial_soc=0.129,
    final_soc=0.90,
    do_plot=True,
):
    energy_demands_per_cluster = {cap: [] for cap in battery_capacities}

    for _ in range(iterations):
        battery_capacity = np.random.choice(battery_capacities, p=cluster_probs)
        initial_soc = np.random.normal(mean_initial_soc, std_dev_initial_soc)
        initial_soc = np.clip(initial_soc, 0, 0.7)

        energy_demand = battery_capacity * (final_soc - initial_soc)
        energy_demands_per_cluster[battery_capacity].append(energy_demand)

    if do_plot:
        plt.figure(figsize=(6, 4))
        hist_colors = ["skyblue", "lightgreen", "salmon"]
        density_colors = ["dodgerblue", "forestgreen", "darkorange"]
        labels = [f"Cluster {i + 1}" for i in range(len(battery_capacities))]

        legend_handles = []

        for i, cap in enumerate(battery_capacities):
            mean_demand = np.mean(energy_demands_per_cluster[cap])
            std_dev_demand = np.std(energy_demands_per_cluster[cap])

            plt.hist(
                energy_demands_per_cluster[cap],
                bins=20,
                density=True,
                alpha=0.5,
                color=hist_colors[i],
                edgecolor="black",
            )

            density = gaussian_kde(energy_demands_per_cluster[cap])
            x_vals = np.linspace(min(energy_demands_per_cluster[cap]),
                                 max(energy_demands_per_cluster[cap]), 1000)
            plt.plot(x_vals, density(x_vals), color=density_colors[i], linewidth=3.5)

            legend_handles.append(
                mlines.Line2D(
                    [], [], color=hist_colors[i], lw=6,
                    label=f"{labels[i]}: Mean={mean_demand:.2f} kWh, Std={std_dev_demand:.2f} kWh"
                )
            )

        plt.xlabel("Energy Demand (kWh)", family="Arial", fontsize=14, fontweight="bold")
        plt.ylabel("Probability Density", family="Arial", fontsize=14, fontweight="bold")
        plt.xticks(family="Arial", fontsize=12, fontweight="bold")
        plt.yticks(family="Arial", fontsize=12, fontweight="bold")
        plt.legend(handles=legend_handles, prop={"family": "Arial", "weight": "bold", "size": 10}, loc="upper right")
        plt.tight_layout()
        plt.show()

    return energy_demands_per_cluster


# ============================================================
# MAIN RUN (prints outputs for each section)
# ============================================================
if __name__ == "__main__":
    # 1) Build IEEE-33 "system" logically (load list is the system input)
    # 2) Divide into sections already defined

    # 3) Estimate EV counts (per bus, per section) for a chosen penetration
    penetration = 0.3176  # change this to compare scenarios
    households_and_evs = estimate_evs_per_bus(LOADS_IEEE33, penetration=penetration)
    evs_per_section, evs_per_bus = aggregate_evs_by_section(households_and_evs, SECTIONS)

    print("\n--- Estimated EV counts by Section ---")
    for sec, n in evs_per_section.items():
        print(f"{sec}: {n} EVs")

    print("\n--- Estimated EV counts by Bus (IEEE-33) ---")
    for b in sorted(evs_per_bus):
        print(f"Bus {b}: {evs_per_bus[b]} EVs")

    # Plot EVs per section (same style as your original)
    plt.figure(figsize=(6, 4))
    plt.bar(evs_per_section.keys(), evs_per_section.values(), color="lightskyblue", edgecolor="black")
    for sec, total_evs in evs_per_section.items():
        plt.text(sec, total_evs - total_evs * 0.1, f"{total_evs:.0f}",
                 ha="center", va="center", fontsize=12, fontweight="bold", color="black")
    plt.ylabel("Total Number of EVs", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.show()

    # 4) Uncertainty plots (optional)
    plot_arrival_time_distribution()
    plot_daily_driven_distance()
    plot_initial_soc_distribution()

    # 5) Plugin interval + daily arrivals per section
    print("\n--- Plugin Intervals + Daily Arrivals (per Section) ---")
    total_arrivals = 0.0
    total_evs_all_sections = sum(evs_per_section.values())

    section_daily_arrivals = []
    for sec_name, total_evs in evs_per_section.items():
        avg_plugin_intervals, avg_daily_arrivals, total_avg_daily_arrival = simulate_section_plugin_intervals(total_evs)

        print(f"\n{sec_name}:")
        print("Average Plugin Interval (Days) per Cluster:")
        for cl, interval in avg_plugin_intervals.items():
            print(f"  {cl}: {interval:.2f} days")
        print("Average Daily Arrival per Cluster:")
        for i, arrival in enumerate(avg_daily_arrivals, start=1):
            print(f"  Cluster {i}: {arrival:.2f} EVs/day")
        print(f"Total Average Daily Arrival: {total_avg_daily_arrival:.2f} EVs/day")

        section_daily_arrivals.append(total_avg_daily_arrival)
        total_arrivals += total_avg_daily_arrival

    percentage_ev_arrival_daily = (total_arrivals / total_evs_all_sections) * 100 if total_evs_all_sections > 0 else 0
    print(f"\nPercentage of EV Arrival Daily: {percentage_ev_arrival_daily:.2f}%")

    # Plot daily arrivals per section
    plt.figure(figsize=(6, 4))
    plt.bar(list(evs_per_section.keys()), section_daily_arrivals, color="lightgreen", edgecolor="black")
    for i, val in enumerate(section_daily_arrivals):
        plt.text(i, val - val * 0.1, f"{val:.0f}", fontsize=14, ha="center", fontweight="bold")
    plt.ylabel("Average daily EV arrival", family="Arial", fontsize=14, fontweight="bold")
    plt.yticks(family="Arial", fontsize=12, fontweight="bold")
    plt.xticks(family="Arial", fontsize=12, fontweight="bold")
    plt.grid(axis="y")
    plt.show()

    # 6) Half-hourly EV demand profiles for each section (48 points)
    print("\n--- Half-hourly EV demand profiles (48 steps) computed ---")
    sec_names = list(evs_per_section.keys())
    demand_48 = {}
    for sec in sec_names:
        demand_48[sec] = half_hourly_section_demand(evs_per_section[sec])

    # Plot half-hourly profiles
    plt.figure(figsize=(6, 4))
    for sec in sec_names:
        plt.plot(range(48), demand_48[sec], linewidth=2.5, label=f"{sec} EV Demand")
    plt.legend(prop={"family": "Arial", "weight": "bold"})
    plt.xlabel("Time (half-hour index)", family="Arial", fontsize=12, fontweight="bold")
    plt.ylabel("EV Power Demand (kW)", family="Arial", fontsize=12, fontweight="bold")
    plt.xticks(fontweight="bold", fontsize=10)
    plt.yticks(fontweight="bold", fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()
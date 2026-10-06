import numpy as np


np.random.seed(42)

N_APPLIANCES = 5
N_SLOTS = 24


appliances = [
    ["Washing Machine", 1.5, 2, 8, 18],
    ["Dishwasher",      1.2, 2, 10, 22],
    ["Water Heater",    2.5, 3, 6, 20],
    ["Air Conditioner", 2.0, 4, 12, 23],
    ["EV Charger",      3.5, 4, 18, 23]
]

price = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.12,
    0.15, 0.18, 0.20, 0.20, 0.18, 0.16,
    0.15, 0.16, 0.18, 0.20, 0.22, 0.25,
    0.30, 0.30, 0.28, 0.20, 0.15, 0.12
])


def create_schedule(start_times):
    """
    Convert each particle position, which stores one start hour per
    appliance, into a 24-hour appliance schedule.
    """
    schedule = np.zeros((N_APPLIANCES, N_SLOTS))

    for i, appliance in enumerate(appliances):
        power = appliance[1]
        duration = appliance[2]
        earliest = appliance[3]
        latest = appliance[4]

        # Latest permissible start hour
        max_start = latest - duration + 1

        # Round the continuous PSO position to a valid integer start hour
        start = int(np.clip(round(start_times[i]), earliest, max_start))

        # Assign appliance power for its required duration
        schedule[i, start:start + duration] = power

    return schedule


def fitness(start_times):
    """
    Objective:
    minimize electricity cost + peak-load penalty.
    """
    schedule = create_schedule(start_times)

    hourly_load = schedule.sum(axis=0)

    energy_cost = np.sum(hourly_load * price)
    peak_load = np.max(hourly_load)
    peak_penalty = 0.05 * peak_load ** 2

    return energy_cost + peak_penalty



NUM_PARTICLES = 60
MAX_ITERATIONS = 200

w_start = 0.7     
w_end = 0.4       
c1 = 1.5           
c2 = 1.5          


lower_bound = np.array([appliance[3] for appliance in appliances])
upper_bound = np.array([
    appliance[4] - appliance[2] + 1
    for appliance in appliances
])


positions = np.random.uniform(
    lower_bound,
    upper_bound,
    (NUM_PARTICLES, N_APPLIANCES)
)

velocities = np.zeros_like(positions)

personal_best_positions = positions.copy()
personal_best_scores = np.array([
    fitness(position) for position in positions
])

best_index = np.argmin(personal_best_scores)
global_best_position = personal_best_positions[best_index].copy()
global_best_score = personal_best_scores[best_index]



for iteration in range(MAX_ITERATIONS):

  
   
    inertia = w_start - (w_start - w_end) * iteration / MAX_ITERATIONS

    r1 = np.random.random((NUM_PARTICLES, N_APPLIANCES))
    r2 = np.random.random((NUM_PARTICLES, N_APPLIANCES))

    velocities = (
        inertia * velocities
        + c1 * r1 * (personal_best_positions - positions)
        + c2 * r2 * (global_best_position - positions)
    )

    positions = np.clip(
        positions + velocities,
        lower_bound,
        upper_bound
    )

   
    current_scores = np.array([
        fitness(position) for position in positions
    ])

  
    improved = current_scores < personal_best_scores
    personal_best_scores[improved] = current_scores[improved]
    personal_best_positions[improved] = positions[improved]
 
    best_index = np.argmin(personal_best_scores)
    if personal_best_scores[best_index] < global_best_score:
        global_best_score = personal_best_scores[best_index]
        global_best_position = personal_best_positions[best_index].copy()



best_schedule = create_schedule(global_best_position)
hourly_load = best_schedule.sum(axis=0)
energy_cost = np.sum(hourly_load * price)
peak_load = np.max(hourly_load)

print("OPTIMAL APPLIANCE SCHEDULE")
for i, appliance in enumerate(appliances):
    operating_hours = np.where(best_schedule[i] > 0)[0]
    print(f"{appliance[0]}: {list(operating_hours)}")

print("\nHourly Power Consumption")
for hour in range(N_SLOTS):
    print(f"Hour {hour:02d}:00 - {hourly_load[hour]:.2f} kW")

print("\nEnergy Cost:", round(energy_cost, 4))
print("Peak Load:", round(peak_load, 2), "kW")
print("Minimum Objective Cost:", round(global_best_score, 4))

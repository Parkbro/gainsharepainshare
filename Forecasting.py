
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


overall_margin = [.07]
gainshare_percentage = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
painshare_percentage = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

variations = []
for om in overall_margin:
    for gp in gainshare_percentage:
        for pp in painshare_percentage:
            variations.append((om, gp, pp))


def calc(
    overall_margin,
    gainshare_percentage,
    painshare_percentage,
    low=85,
    high=115,
    confidence=.85,
    gainshare_interval=3,
    number_of_sims=1_000
):
    # Establish a Budget for Expenditure
    purchase_values = np.array(np.random.randint(100, 200, 10))
    annual_values = purchase_values * (1 + overall_margin)
    # Set how often the gainshare is
    gainshare_every_x_years = gainshare_interval
    # Calculate the random norm distributed percentages
    random_flux = np.array([
        # np.random.normal(1.0, flux_amount, len(annual_values))
        np.random.randint(low, high, len(annual_values)) / 100
        for n in range(number_of_sims)
    ])
    # Multiple percentage arrays vs value array
    simulated_values = annual_values * random_flux

    # Allow slicing of array into chunks for cumulative summing operations
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # Calculate an Boolean array for whether it is a gainshare year
    gainshare_map = np.array(
        [1 if x % gainshare_every_x_years == 0 else 0
            for x in range(1, len(annual_values) + 1)
         ]
    )
    # Calculate the rolling budget for which the gainshare will be calculated
    # Check contractual terms for how the gainshare is split (there might be
    # a cushion on annual expenditure if it can roll into the next year)
    rolling_cumultive_budget = np.cumsum(
        rolling_window(
            annual_values, gainshare_every_x_years
        ), -1
    )
    # Calculate Simulated Rolling Totals
    rolling_cumulative_sum = np.cumsum(
        rolling_window(
            simulated_values, gainshare_every_x_years
        ), -1
    )

    # Create a dictionary container to hold simulation results
    results = {}

    # zip simulations and their corresponding cumulative sums
    # use the index to create unique dictionary keys
    for idx, (sv, rs) in enumerate(
            zip(simulated_values, rolling_cumulative_sum),
            1):

        # Create individual dictionary
        results[str(idx)] = {}
        """
        Calculate Profit by Condition:

        If simulated values exceed the budget for that year take the budget and
        subtract the annual values - purchase values because you can only make
        profit up to the price ceiling

        If simulated values are below or equal to annual budget, take the
        actual simulated budget expenditure and multiply it by the overall
        margin that was agreed
        """
        results[str(idx)]["profit"] = (
            (sv > annual_values).astype(int) * (
                annual_values - purchase_values  # This is the 7% margin
            )
        ) + (
            (sv <= annual_values).astype(int) * (sv * (overall_margin))
            # This is the 7% margin on what was actually purchased
        )

        #  Create some containers to store the results
        results[str(idx)]["gainshare"] = []
        results[str(idx)]["painshare"] = []
        results[str(idx)]["pc_gainshare"] = []
        results[str(idx)]["pc_painshare"] = []

        #  Calculate the gainshare/painshare conditionally on the index
        for sim, budget, m in zip(rs, rolling_cumultive_budget, gainshare_map):
            # m means it's a gainshare year
            interval = gainshare_every_x_years - 1

            if m == 1:
                # If the simulated total is lower than the budget,
                # there is a gainshare and no painshare
                if sim.item(interval) <= budget.item(interval):
                    results[str(idx)]["painshare"].append(0.0)
                    results[str(idx)]["pc_painshare"].append(0.0)
                    # Gainshare percentage to subbie
                    results[str(idx)]["gainshare"].append(
                        (budget.item(interval) - sim.item(interval)) *
                        gainshare_percentage
                    )
                    # Gainshare percentage to primary is 1 - the subbies
                    results[str(idx)]["pc_gainshare"].append(
                        (budget.item(interval) - sim.item(interval)) *
                        (1.0 - gainshare_percentage)
                    )
                # The case of where the simulated total exceeds the budget
                # cumulatively during the end of gainshare period
                else:
                    # Both parties get no gain obviously
                    results[str(idx)]["gainshare"].append(0.0)
                    results[str(idx)]["pc_gainshare"].append(0.0)
                    # Sub takes amount exceeded * percetage of painshare
                    results[str(idx)]["painshare"].append(
                        (sim.item(interval) - budget.item(interval)) *
                        painshare_percentage
                    )
                    # Primary takes amount exceeded * percentage of painshare
                    results[str(idx)]["pc_painshare"].append(
                        (sim.item(interval) - budget.item(interval)) *
                        (1.0 - painshare_percentage)
                    )
            else:
                # Otherwise there is no gainshare/painshare because
                # it is not a gainshare year we are calculating
                results[str(idx)]["gainshare"].append(0.0)
                results[str(idx)]["painshare"].append(0.0)
                results[str(idx)]["pc_gainshare"].append(0.0)
                results[str(idx)]["pc_painshare"].append(0.0)

        #  Transform the lists into Numpy arrays so we can do calculations
        #  on the vectors instead of line by line.
        results[str(idx)]["gainshare"] = np.array(
            results[str(idx)]["gainshare"]
        )

        results[str(idx)]["painshare"] = np.array(
            results[str(idx)]["painshare"]
        )

        results[str(idx)]["pc_gainshare"] = np.array(
            results[str(idx)]["pc_gainshare"]
        )

        results[str(idx)]["pc_painshare"] = np.array(
            results[str(idx)]["pc_painshare"]
        )

        results[str(idx)]["break_even"] = np.sum(
            results[str(idx)]["gainshare"] - results[str(idx)]["painshare"]
        )

        results[str(idx)]["margin"] = np.sum(
            results[str(idx)]["profit"]
        ) + results[str(idx)]["break_even"]

        results[str(idx)]["margin_pct"] = results[str(
            idx)]["margin"] / np.sum(annual_values)

        results[str(idx)]["pc_margin_pct"] = np.sum(
            results[str(idx)]["pc_gainshare"] -
            results[str(idx)]["pc_painshare"]
        ) / np.sum(annual_values)

    totals = []
    pc_totals = []
    for k, v in results.items():
        totals.append(v["margin_pct"])
        pc_totals.append(v["pc_margin_pct"])

    # totals.sort()
    # pc_totals.sort()
    sorted_together = np.rec.fromarrays([totals, pc_totals])
    # print(np.correlate(sorted_together.f0, sorted_together.f1))
    sorted_together.sort()

    return (
        overall_margin,
        gainshare_percentage,
        painshare_percentage,
        # Following our coversation we will take the median
        np.median(sorted_together.f0) * 100,
        # This will have to be the confidence interval
        np.median(totals) * 100,
        np.median(sorted_together.f1) * 100
    )


# These are the random number +/- amounts
t_b = [(75, 125), (85, 110), (90, 110), (90, 120)]

df_results = {}

for top, bottom in t_b:
    col_margin = []
    col_gain = []
    col_pain = []
    col_mean = []
    col_median = []
    col_pc_tot = []

    for a, budget, c in variations:
        a = calc(a, budget, c, top, bottom,
                 confidence=.95, number_of_sims=750)
        col_margin.append(a[0])
        col_gain.append(a[1])
        col_pain.append(a[2])
        col_mean.append(a[3])
        col_median.append(a[4])
        col_pc_tot.append(a[5])

    df_results[
        "df-{}% to {}%".format(top - 100, bottom - 100)
    ] = pd.DataFrame.from_dict(
        {"margin": col_margin,
         "gain": col_gain,
         "pain": col_pain,
         "mean": col_mean,
         "median": col_median,
         "pc_mean": col_pc_tot}
    )

labels = list(df_results.keys())

df = df_results[labels[0]]
df.name = labels[0].replace("df-", "")
df2 = df_results[labels[1]]
df2.name = labels[1].replace("df-", "")
df3 = df_results[labels[2]]
df3.name = labels[2].replace("df-", "")
df4 = df_results[labels[3]]
df4.name = labels[3].replace("df-", "")

custom_axis_down = [
    "100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%", "0%"
]
custom_axis_up = custom_axis_down[:]
# Does it in place
custom_axis_up.reverse()

f, axarr = plt.subplots(2, 4)

axarr[0, 0].set_title("Subcontractor: {}".format(df4.name))
sns.heatmap(df4.pivot("gain", "pain", "mean").round(1),
            annot=True, cbar=False, ax=axarr[0, 0],
            yticklabels=custom_axis_up, xticklabels=custom_axis_up,
            cmap="coolwarm")

axarr[0, 1].set_title("Subcontractor: {}".format(df2.name))
sns.heatmap(df2.pivot("gain", "pain", "mean").round(1),
            annot=True, cbar=False, ax=axarr[0, 1],
            yticklabels=custom_axis_up, xticklabels=custom_axis_up,
            cmap="coolwarm")

axarr[0, 2].set_title("Subcontractor: {}".format(df3.name))
sns.heatmap(df3.pivot("gain", "pain", "mean").round(1),
            annot=True, cbar=False, ax=axarr[0, 2],
            yticklabels=custom_axis_up, xticklabels=custom_axis_up,
            cmap="coolwarm")

axarr[0, 3].set_title("Subcontractor: {}".format(df.name))
sns.heatmap(df.pivot("gain", "pain", "mean").round(1),
            annot=True, cbar=False, ax=axarr[0, 3],
            yticklabels=custom_axis_up, xticklabels=custom_axis_up,
            cmap="coolwarm")

axarr[1, 0].set_title("Primary Contractor: {}".format(df4.name))
sns.heatmap(df4.pivot("gain", "pain", "pc_mean").round(1),
            annot=True, cbar=False, ax=axarr[1, 0],
            yticklabels=custom_axis_down, xticklabels=custom_axis_down)

axarr[1, 1].set_title("Primary Contractor: {}".format(df2.name))
sns.heatmap(df2.pivot("gain", "pain", "pc_mean").round(1),
            annot=True, cbar=False, ax=axarr[1, 1],
            yticklabels=custom_axis_down, xticklabels=custom_axis_down)

axarr[1, 2].set_title("Primary Contractor: {}".format(df3.name))
sns.heatmap(df3.pivot("gain", "pain", "pc_mean").round(1),
            annot=True, cbar=False, ax=axarr[1, 2],
            yticklabels=custom_axis_down, xticklabels=custom_axis_down)

axarr[1, 3].set_title("Primary Contractor: {}".format(df.name))
sns.heatmap(df.pivot("gain", "pain", "pc_mean").round(1),
            annot=True, cbar=False, ax=axarr[1, 3],
            yticklabels=custom_axis_down, xticklabels=custom_axis_down)

f.subplots_adjust(hspace=.3, wspace=.2)

plt.show()

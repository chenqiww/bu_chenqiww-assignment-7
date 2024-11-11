from flask import Flask, render_template, request, url_for, session, flash, redirect
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

# Ensure the 'static' directory exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

def generate_data(N, mu, beta0, beta1, sigma2, S, seed):
    # Set the random seed
    np.random.seed(seed)

    # Generate X and epsilon
    X = np.random.uniform(0, 1, N)
    epsilon = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + epsilon  # Corrected to add mu separately

    # Fit linear regression model
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate scatter plot with regression line
    plot1_path = "static/plot1.png"
    fig, ax = plt.subplots()
    ax.scatter(X, Y, label='Data Points', alpha=0.6)
    X_sorted = np.sort(X)
    X_sorted_reshaped = X_sorted.reshape(-1, 1)
    Y_pred_sorted = model.predict(X_sorted_reshaped)
    ax.plot(X_sorted, Y_pred_sorted, color='red', label='Fitted Line')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Linear Fit: Y = {intercept:.4f} + {slope:.4f}X')
    ax.legend()
    plt.savefig(plot1_path)
    plt.close(fig)

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        epsilon_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + epsilon_sim

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts on the same axes
    plot2_path = "static/plot2.png"
    fig2, ax2 = plt.subplots()
    bins = 30  # Number of bins

    # Determine common bin edges
    all_values = np.concatenate((slopes, intercepts))
    bin_edges = np.linspace(min(all_values), max(all_values), bins + 1)

    # Plot histogram for Intercepts first (lighter color)
    ax2.hist(intercepts, bins=bin_edges, alpha=0.5, label='Intercepts', color='orange', edgecolor='black')

    # Plot histogram for Slopes
    ax2.hist(slopes, bins=bin_edges, alpha=0.5, label='Slopes', color='blue', edgecolor='black')

    # Plot vertical lines for observed slope and intercept
    ax2.axvline(slope, color='blue', linestyle='solid', linewidth=2, label=f'Observed Slope: {slope:.4f}')
    ax2.axvline(intercept, color='orange', linestyle='dashed', linewidth=2, label=f'Observed Intercept: {intercept:.4f}')

    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Slopes and Intercepts')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close(fig2)

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Return data needed for further analysis, including simulations
    return (
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        try:
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            beta0 = float(request.form["beta0"])
            beta1 = float(request.form["beta1"])
            S = int(request.form["S"])

            # Input validation
            N_max = 10000
            S_max = 10000
            error_message = None
            if N <= 0 or N > N_max:
                error_message = f"Sample size N must be a positive integer less than or equal to {N_max}."
            elif S <= 0 or S > S_max:
                error_message = f"Number of simulations S must be a positive integer less than or equal to {S_max}."
            elif sigma2 < 0:
                error_message = "Variance sigma² must be non-negative."

            if error_message:
                flash(error_message)
                return redirect(url_for('index'))

            # Generate a random seed
            seed = np.random.randint(0, 1000000)
            # Store the seed in session
            session['seed'] = seed

            # Generate data and initial plots
            (
                slope,
                intercept,
                plot1,
                plot2,
                slope_extreme,
                intercept_extreme,
                slopes,
                intercepts
            ) = generate_data(N, mu, beta0, beta1, sigma2, S, seed)

            # Store data in session
            session['params'] = {
                'N': N,
                'mu': mu,
                'sigma2': sigma2,
                'beta0': beta0,
                'beta1': beta1,
                'S': S,
                'observed_slope': slope,
                'observed_intercept': intercept,
                'slope_extreme': slope_extreme,
                'intercept_extreme': intercept_extreme
            }

            # Store simulations in session
            session['slopes'] = slopes
            session['intercepts'] = intercepts

            # Return render_template with variables
            return render_template(
                "index.html",
                plot1=True,
                plot2=True,
                slope_extreme=slope_extreme,
                intercept_extreme=intercept_extreme,
                N=N,
                mu=mu,
                sigma2=sigma2,
                beta0=beta0,
                beta1=beta1,
                S=S,
            )
        except ValueError:
            error_message = "Please enter valid numeric values for all fields."
            flash(error_message)
            return redirect(url_for('index'))
    else:
        # Set default values
        N = 100
        mu = 0
        sigma2 = 1
        beta0 = 0   # Default value for beta0
        beta1 = 1   # Default value for beta1
        S = 1000
        return render_template(
            "index.html",
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S
        )

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    params = session.get('params')
    seed = session.get('seed')
    slopes = session.get('slopes')
    intercepts = session.get('intercepts')
    if not params or seed is None or slopes is None or intercepts is None:
        flash("Please generate data before running hypothesis testing.")
        return redirect(url_for('index'))

    N = params['N']
    mu = params['mu']
    sigma2 = params['sigma2']
    beta0 = params['beta0']
    beta1 = params['beta1']
    S = params['S']
    observed_slope = params['observed_slope']
    observed_intercept = params['observed_intercept']

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if not parameter or not test_type:
        flash("Please select a parameter and test type for hypothesis testing.")
        return redirect(url_for('index'))

    # Use the stored simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = observed_slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = observed_intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == '>':
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == '<':
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # '!='
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    # If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value <= 0.0001:
        fun_message = "Wow! You've encountered a rare event with a very small p-value!"
    else:
        fun_message = None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    fig3, ax3 = plt.subplots()
    ax3.hist(simulated_stats, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Simulated Statistics')
    ax3.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label=f'Observed {parameter.capitalize()}: {observed_stat:.4f}')
    ax3.axvline(hypothesized_value, color='green', linestyle='solid', linewidth=2, label=f'Hypothesized {parameter.capitalize()} (H₀): {hypothesized_value:.4f}')
    ax3.set_xlabel(f'{parameter.capitalize()}')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Hypothesis Test for {parameter.capitalize()}')
    ax3.legend(loc='upper left')
    plt.savefig(plot3_path)
    plt.close(fig3)

    # Return results to template
    return render_template(
        "index.html",
        plot1=True,
        plot2=True,
        plot3=True,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    params = session.get('params')
    seed = session.get('seed')
    slopes = session.get('slopes')
    intercepts = session.get('intercepts')
    if not params or seed is None or slopes is None or intercepts is None:
        flash("Please generate data before calculating confidence intervals.")
        return redirect(url_for('index'))

    N = params['N']
    mu = params['mu']
    sigma2 = params['sigma2']
    beta0 = params['beta0']
    beta1 = params['beta1']
    S = params['S']
    observed_slope = params['observed_slope']
    observed_intercept = params['observed_intercept']

    parameter = request.form.get("parameter")
    confidence_level_str = request.form.get("confidence_level")
    if not parameter or not confidence_level_str:
        flash("Please select a parameter and confidence level for the confidence interval.")
        return redirect(url_for('index'))

    confidence_level = float(confidence_level_str)

    # Use the stored simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = observed_slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = observed_intercept
        true_param = beta0

    # Calculate mean and standard error
    mean_estimate = np.mean(estimates)
    std_error = stats.sem(estimates)

    # Degrees of freedom
    df = len(estimates) - 1

    # t critical value
    t_crit = stats.t.ppf((1 + confidence_level / 100) / 2, df)

    # Confidence interval
    ci_lower = mean_estimate - t_crit * std_error
    ci_upper = mean_estimate + t_crit * std_error

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates and confidence interval
    plot4_path = "static/plot4.png"
    fig4, ax4 = plt.subplots()
    ax4.scatter(estimates, np.zeros_like(estimates), color='gray', alpha=0.5, label='Simulated Estimates')
    # Determine color based on inclusion of true parameter
    ci_color = 'green' if includes_true else 'red'
    # Plot the confidence interval
    ax4.hlines(y=0, xmin=ci_lower, xmax=ci_upper, colors=ci_color, linestyles='solid', linewidth=4, label=f'{confidence_level}% Confidence Interval')
    # Plot the mean estimate
    ax4.plot(mean_estimate, 0, 'o', color='orange', label='Mean Estimate')
    # Plot the true parameter
    ax4.axvline(true_param, color='blue', linestyle='dashed', linewidth=2, label='True Parameter')
    ax4.set_xlabel(f'{parameter.capitalize()} Estimate')
    ax4.set_yticks([])
    ax4.set_title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()} (t-distribution)')
    ax4.legend(loc='upper right')
    plt.savefig(plot4_path)
    plt.close(fig4)

    # Return results to template
    return render_template(
        "index.html",
        plot1=True,
        plot2=True,
        plot4=True,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True)

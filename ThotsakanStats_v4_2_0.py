#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re # <-- ADDED IMPORT
import numpy as np
import pandas as pd
import tempfile

# To plot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# To create GUI
import gradio as gr

# To integrate a function
from scipy.integrate import quad, dblquad

# Gamma function
from scipy.special import gamma, loggamma

# To calculate statistics
from scipy.stats import norm, t, chi2
from scipy.stats import hmean, trim_mean, iqr, median_abs_deviation, skew, kurtosis
from scipy.stats.mstats import gmean, winsorize
from statsmodels.distributions import ECDF

# To make hypothesis testing
import pingouin as pg
from scipy.stats import bartlett, levene
from scipy.stats import gaussian_kde

# To make linear regression
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


ROUND = 4 # Number of decimals to round the results

df_cache = {
    "df": None,
    "filtered_df": None,
    "stats": None,
    "numeric_cols": [],
    "categorical_cols": [],
    "overrides": {
        "num_to_cat": [],
        "cat_to_num": []
    }
}

export_cache = {
    "table": None,
    "figure": None
}


# # 🧠 📐 📊 🧮 💭 Statistics class

# In[3]:


class Statistics():
    def __init__(self, data):
        self.data = data    
        self.n = len(data)  

    # --- Descriptive Statistics ---    

    # --- Quantiles ---
    def CalculateQuantiles(self, prob):
        prob = np.atleast_1d(prob)
        self.quantiles = pd.DataFrame({"Value": np.quantile(self.data, prob)}, [f"Q{p}" for p in prob])

    # --- Quartiles ---
    def CalculateQuartiles(self):
        self.quartiles = pd.DataFrame({'Value': np.quantile(self.data, [0.25,0.5,0.75])},['Q1', 'Q2', 'Q3'])

    # --- Central Tendency ---
    def CalculateCentralTendency(self, trim_param=0.1, winsor_param=[0.1,0.1], weights=None):

        #Mode = mode(data) # To calculate the mode 
        self.mean = self.data.mean()
        self.median = np.median(self.data)
        self.interquartile_mean = trim_mean(self.data, 0.25)
        
        if trim_param is None:
            self.trim_param = None
            self.trimmed_mean = np.nan
        else:
            self.trim_param = trim_param
            self.trimmed_mean = trim_mean(self.data, trim_param)

        if winsor_param is None:
            self.winsor_param = None
            self.winsorized_mean = np.nan
        else:
            # Winsorized data
            self.data_winsorized = winsorize(self.data, winsor_param)
            self.winsorized_mean = self.data_winsorized.mean()
            
        if np.all(self.data > 0): # If all observations are greater than zero, calculate geometric and harmonic mean
            self.geometric_mean = gmean(self.data)
            self.harmonic_mean = hmean(self.data)
        else:
            self.geometric_mean = np.nan
            self.harmonic_mean = np.nan

        if weights is None:
            self.weighted_mean = np.nan
        else:
            self.weighted_mean = np.average(self.data, weights=weights)

        # Write the statistics in a list
        central_tendency = [
            self.mean,
            self.median,
            self.geometric_mean,
            self.harmonic_mean,
            self.weighted_mean,
            self.trimmed_mean,
            self.interquartile_mean,
            self.winsorized_mean
        ]

        # Return the statistics as a table
        labels = ['Mean', 'Median', 'Geometric Mean', 'Harmonic Mean', 'Weighted Mean', 'Trimmed Mean', 'Interquartile Mean', 'Winsorized Mean']
        self.central_tendency = pd.DataFrame({'Value':central_tendency, 'Robust':[0, 1, 0, 0, 0, 1, 1, 1]}, labels)

    # --- Dispersion ---
    # Auxiliary functions to correct the bias
    def c4(self, n):
        return np.exp(np.log(np.sqrt(2/(n-1))) + loggamma(n/2) - loggamma((n-1)/2))
        #return np.sqrt(2/(n-1)) * gamma(n/2) / gamma((n-1)/2)

    def d2(self, n):
        f = lambda x, n: 1 - (1 - norm.cdf(x))**n - (norm.cdf(x))**n
        return round(quad(f, -np.inf, np.inf, args=(n,))[0], 3)

    def CalculateDispersion(self):
        # Original estimators
        self.S0 = np.std(self.data)             # By default the standard deviation is calculated with zero degrees of freedom
        self.S1 = np.std(self.data, ddof=1)     # Standard deviation with one degree of freedom
        self.R = self.data.max() - self.data.min()
        self.IQR = iqr(self.data)                
        self.MAD = median_abs_deviation(self.data)   
        self.AAD = abs(self.data - self.data.mean()).mean()

        # Bias correction
        self.S0_bias_correct = self.S0 * np.sqrt(self.n/(self.n-1)) / self.c4(self.n)
        self.S1_bias_corrected = self.S1 / self.c4(self.n)
        self.R_bias_corrected = self.R / self.d2(self.n)
        self.IQR_bias_corrected = self.IQR / (2 * norm.ppf(0.75))
        self.MAD_bias_corrected = self.MAD / norm.ppf(0.75)
        self.AAD_bias_corrected = self.AAD * np.sqrt(np.pi/2)

        # Write the statistics in a list
        sigma_biased = [self.S0, self.S1, self.R, self.IQR, self.MAD, self.AAD]
        sigma_unbiased = [
            self.S0_bias_correct,
            self.S1_bias_corrected,
            self.R_bias_corrected,
            self.IQR_bias_corrected,
            self.MAD_bias_corrected,
            self.AAD_bias_corrected
        ] 

        # Return the statistics as a table
        labels = ['Deviation, ddof=0', 'Deviation, ddof=1', 'Range', 'IQR', 'MAD', 'AAD']
        self.dispersion = pd.DataFrame({'Value':sigma_biased, 'Value_bias_corrected':sigma_unbiased, 'Robust':[0,0,0,1,1,1]}, labels)

    # --- Skew ---
    def CalculateSkewness(self):
        SkewCentralMoments = skew(self.data)
        SkewKStatistics = skew(self.data, bias=False)

        self.skew = pd.DataFrame({'Value':[SkewCentralMoments, SkewKStatistics]}, ['Skew Central Moments', 'Skew K Statistics'])
    
    # --- Kurtosis ---
    def CalculateKurtosis(self):
        KurtosisCentralMoments = kurtosis(self.data, fisher=False)
        KurtosisKStatistics = kurtosis(self.data, fisher=False, bias=False)

        self.kurtosis = pd.DataFrame(
            {'Value':[KurtosisCentralMoments, KurtosisKStatistics], 'Excess Kurtosis':[KurtosisCentralMoments-3, KurtosisKStatistics-3]},
            ['Kurtosis Central Moments', 'Kurtosis K Statistics']
        )

    def CalculateDescriptiveStatistics(self, trim_param, winsor_param, weights):
        self.CalculateQuartiles()
        self.CalculateCentralTendency(trim_param, winsor_param, weights)
        self.CalculateDispersion()
        self.CalculateSkewness()
        self.CalculateKurtosis()

    # --- Statistical Inference ---

    # --- Confidence Intervals ---
    def CalculateCiMean(self, alpha, hat_mean, hat_sigma, dist, sel, boots, bootstrap_samples):

        if boots:
            if sel == "Sample Mean":
                boot_means = np.array([
                    np.mean(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])
            elif sel == "Geometric Mean":
                if np.all(self.data > 0):
                    boot_means = np.array([
                        gmean(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])
                else:
                    self.ci_mean = np.nan, np.nan
            elif sel == "Harmonic Mean":
                if np.all(self.data > 0):
                    boot_means = np.array([
                        hmean(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])
                else:
                    self.ci_mean = np.nan, np.nan
            elif sel == "Trimmed Mean":
                if self.trim_param is None:
                    self.ci_mean = np.nan, np.nan
                else:
                    boot_means = np.array([
                        trim_mean(np.random.choice(self.data, size=self.n, replace=True), self.trim_param) for _ in range(bootstrap_samples)
                    ])
            elif sel == "Interquartile Mean":
                boot_means = np.array([
                    trim_mean(np.random.choice(self.data, size=self.n, replace=True), 0.25) for _ in range(bootstrap_samples)
                ])
            elif sel == "Winsorized Mean":
                if self.winsor_param is None:
                    return np.nan, np.nan
                else:
                    boot_means = np.array([
                        np.mean(np.random.choice(self.data_winsorized, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])
            self.ci_mean = np.quantile(boot_means, [alpha/2, 1-alpha/2])
        else:
            # Calculate confidence interval for the mean
            scale = hat_sigma / np.sqrt(self.n)

            if dist=="norm":
                self.ci_mean = norm.ppf(alpha/2, hat_mean, scale), norm.ppf(1-alpha/2, hat_mean, scale)
            if dist=="t":
                # Only if we are using standard deviaiton with one degree of freedom without correction
                self.ci_mean = t.ppf(alpha/2, self.n-1, hat_mean, scale), t.ppf(1-alpha/2, self.n-1, hat_mean, scale)
        
    def CalculateCiMedian(self, alpha, hat_median, hat_sigma, boots, bootstrap_samples):
        if boots:
            boot_medians = np.array([
                np.median(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
            ])

            self.ci_median = np.quantile(boot_medians, [alpha/2, 1-alpha/2])
        else:
            # Calculate confidence interval based on the median
            scale = hat_sigma * np.sqrt(np.pi/(2*self.n))
            self.ci_median = norm.ppf(alpha/2, hat_median, scale), norm.ppf(1-alpha/2, hat_median, scale)

    def d3(self, n):
        d3_table = {
            2: 0.852, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848,
            7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797,
            11: 0.787, 12: 0.778, 13: 0.770, 14: 0.763, 15: 0.756,
            16: 0.750, 17: 0.744, 18: 0.739, 19: 0.734, 20: 0.729,
            21: 0.724, 22: 0.720, 23: 0.716, 24: 0.712, 25: 0.708
        }
        return d3_table[n]

    def d3_exact(self, n):
        """Compute the d3 constant (stddev of range) for sample size n of normal distribution."""
        # PDF and CDF of standard normal
        phi = norm.pdf
        Phi = norm.cdf

        # Expected range squared E[R_n^2]
        def integrand2(x2, x1):
            return (x2 - x1)**2 * phi(x1) * phi(x2) * (Phi(x2) - Phi(x1))**(n-2)
        
        ERn2, _ = dblquad(
            integrand2,
            -np.inf, np.inf,
            lambda x1: x1,
            lambda x1: np.inf,
            epsabs=1e-10, epsrel=1e-8
        )
        ERn2 = n * (n - 1) * ERn2
        
        ERn = self.d2(n)
        var_Rn = ERn2 - ERn**2
        return np.sqrt(var_Rn)

    def ci_sigma_from_range(self, alpha):
        if self.n <= 25:
            d2_n = self.d2(self.n)
            d3_n = self.d3(self.n)
        else:
            d2_n = self.d2(self.n)
            d3_n = self.d3_exact(self.n)

        z = norm.ppf(1 - alpha / 2)
        
        denom_lo = d2_n + z * d3_n
        denom_hi = d2_n - z * d3_n
        
        if denom_hi <= 0:
            raise ValueError("Invalid configuration: upper bound denominator ≤ 0")
        
        return self.R / denom_lo, self.R / denom_hi

    def CalculateCiDeviation(self, alpha, sel, boots, bootstrap_samples):
        if boots:
            if sel == "Deviation (1 ddof)":
                boot_deviations = np.array([
                        np.std(np.random.choice(self.data, size=self.n, replace=True), ddof=1) for _ in range(bootstrap_samples)
                    ])

                self.ci_deviation = np.quantile(boot_deviations, [alpha/2, 1-alpha/2]) / self.c4(self.n)
                #self.S1 = np.quantile(boot_deviations, 0.5)
                #self.S1_bias_corrected = self.S1 / self.c4(self.n)

                #self.S0_bias_corrected = self.S1_bias_corrected
                #self.S0 = np.sqrt(self.n/(self.n-1))

            elif sel == "Range (bias corrected)":
                boot_deviations = []
                for _ in range(bootstrap_samples):
                    boot_data = np.random.choice(self.data, size=self.n, replace=True)
                    deviation = boot_data.max() - boot_data.min()
                    boot_deviations.append(deviation)
                boot_deviations = np.array(boot_deviations)

                self.ci_deviation = np.quantile(boot_deviations, [alpha/2, 1-alpha/2]) / self.d2(self.n)
                #self.R = np.quantile(boot_deviations, 0.5)
                #self.R_bias_corrected = self.R / self.d2(self.n)

            elif sel == "IQR (bias corrected)":
                boot_deviations = np.array([
                        iqr(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])

                self.ci_deviation = np.quantile(boot_deviations, [alpha/2, 1-alpha/2]) / (2 * norm.ppf(0.75))
                #self.IQR = np.quantile(boot_deviations, 0.5)
                #self.IQR_bias_corrected = self.IQR / (2 * norm.ppf(0.75))

            elif sel == "MAD (bias corrected)":
                boot_deviations = np.array([
                        median_abs_deviation(np.random.choice(self.data, size=self.n, replace=True)) for _ in range(bootstrap_samples)
                    ])

                self.ci_deviation = np.quantile(boot_deviations, [alpha/2, 1-alpha/2]) / (norm.ppf(0.75))
                #self.MAD = np.quantile(boot_deviations, 0.5)
                #self.MAD_bias_corrected = self.MAD / (norm.ppf(0.75))

            elif sel == "AAD (bias corrected)":
                boot_deviations = []
                for _ in range(bootstrap_samples):
                    boot_data = np.random.choice(self.data, size=self.n, replace=True)
                    deviation = abs(boot_data - boot_data.mean()).mean()
                    boot_deviations.append(deviation)
                boot_deviations = np.array(boot_deviations)

                self.ci_deviation = np.quantile(boot_deviations, [alpha/2, 1-alpha/2]) * np.sqrt(np.pi/2)
                #self.AAD = np.quantile(boot_deviations, 0.5)
                #self.AAD_bias_corrected = self.AAD * np.sqrt(np.pi/2)
        
        else:
            if sel == "Deviation (1 ddof)":
                # Exact Distribution
                num = self.S1 * np.sqrt(self.n-1)
                den_low = np.sqrt(chi2.ppf(1-alpha/2, self.n-1))
                den_upp = np.sqrt(chi2.ppf(alpha/2, self.n-1))

                self.ci_deviation = num/den_low, num/den_upp
            elif sel == "Range (bias corrected)":
                # Exact Distribution
                self.ci_deviation = self.ci_sigma_from_range(alpha)
            elif sel == "IQR (bias corrected)":
                # Asymptotic Distribution
                w = 2 * norm.ppf(0.75)
                k = np.sqrt(np.pi/(2*np.exp(-norm.ppf(0.75)**2)))
                z = norm.ppf(1-alpha/2)
                self.ci_deviation = self.IQR / (w+z*k/np.sqrt(self.n)), self.IQR / (w-z*k/np.sqrt(self.n))
            elif sel == "MAD (bias corrected)":
                # Asymptotic Distribution
                w = norm.ppf(0.75)
                k = np.sqrt(np.pi/(8*np.exp(-norm.ppf(0.75)**2)))
                z = norm.ppf(1-alpha/2)
                self.ci_deviation = self.MAD / (w+z*k/np.sqrt(self.n)), self.MAD / (w-z*k/np.sqrt(self.n))
            elif sel == "AAD (bias corrected)":
                # Asymptotic Distribution
                w = np.sqrt(2/np.pi)
                k = np.sqrt((1-2/np.pi))
                z = norm.ppf(1-alpha/2)
                self.ci_deviation = self.AAD / (w+z*k/np.sqrt(self.n)), self.AAD / (w-z*k/np.sqrt(self.n))

    def CalculateConfidenceInterval(
            self, alpha, hat_mean, hat_median, hat_sigma, dist,
            mean_select, sigma_select, boots_mean, boots_median, boots_deviation, bootstrap_samples):
        self.CalculateCiMean(alpha, hat_mean, hat_sigma, dist, mean_select, boots_mean, bootstrap_samples)
        self.CalculateCiMedian(alpha, hat_median, hat_sigma, boots_median, bootstrap_samples)
        self.CalculateCiDeviation(alpha, sigma_select, boots_deviation, bootstrap_samples)

        # Return the statistics as a table
        labels = ['Mean', 'Median', 'Deviation']
        self.confidence_intervals = pd.DataFrame(
            [self.ci_mean, self.ci_median, self.ci_deviation],
            index=labels, columns=["Lower", "Upper"]
        )

    # --- Prediction Intervals ---
    def CalculatePiMean(self, alpha, hat_mean, hat_sigma, dist):
        # Calculate prediction interval based on the mean
        scale = np.sqrt(hat_sigma**2 + hat_sigma**2/self.n)

        if dist == "norm":
            self.pi_mean = norm.ppf(alpha/2, hat_mean, scale), norm.ppf(1-alpha/2, hat_mean, scale)
        if dist == "t":
            # Only if we are using standard deviaiton with one degree of freedom without correction
            self.pi_mean = t.ppf(alpha/2, self.n-1, hat_mean, scale), t.ppf(1-alpha/2, self.n-1, hat_mean, scale)
    
    def CalculatePiMedian(self, alpha, hat_median=None, hat_sigma=None):
        scale = np.sqrt(hat_sigma**2 + np.pi*hat_sigma**2/(2*self.n))
        self.pi_median = norm.ppf(alpha/2, hat_median, scale), norm.ppf(1-alpha/2, hat_median, scale)

    def CalculatePiIqr(self, alpha):
        # Calculate prediction interval based on the first and third quartile
        q1, q3 = np.quantile(self.data, [0.25, 0.75])
        iqr = q3-q1
        delta = 0.5 * (norm.ppf(1-alpha/2)/norm.ppf(0.75)-1)

        self.pi_iqr = q1 - delta * iqr, q3 + delta * iqr

    def CalculatePiBoots(self, alpha, bootstrap_samples):
        labels = ["Bootstrap"]
        self.pi_boots = np.mean(
            np.array([np.quantile(np.random.choice(self.data, size=self.n, replace=True), [alpha/2, 1-alpha/2]) for _ in range(bootstrap_samples)]),
            axis=0)

        self.prediction_intervals_boots = pd.DataFrame([self.pi_boots], index=labels, columns=["Lower", "Upper"])

    def CalculatePredictionInterval(self, alpha, hat_mean, hat_median, hat_sigma, dist):
        self.CalculatePiMean(alpha, hat_mean, hat_sigma, dist)
        self.CalculatePiMedian(alpha, hat_median, hat_sigma)
        self.CalculatePiIqr(alpha)

        # Return the statistics as a table
        labels = ['Mean', 'Median', 'IQR']
        self.prediction_intervals = pd.DataFrame(
            [self.pi_mean, self.pi_median, self.pi_iqr],
            index=labels, columns=["Lower", "Upper"]
        )

    # --- Relative Likelihood ---
    def RelativeLogLikelihood(self, mu, sigma):
        return self.n * (np.log(self.S0 / sigma) + 0.5 * (1 - (np.mean(self.data**2) - 2 * mu * np.mean(self.data) + mu**2) / sigma**2))

    def RelativeLikelihood(self, mu, sigma):
        return np.exp(self.RelativeLogLikelihood(mu, sigma))

    # --- Graphical Analysis ---
    
    # --- Plot Histogram ---
    def PlotHistogram(self, name_variable, kde, show_data, histo_add_ci, histo_choose_ci, histo_add_pi, histo_choose_pi, add_normal, hat_mu, hat_sigma):
        # Style
        plt.style.use("seaborn-v0_8-whitegrid")
        
        show_intervals = histo_add_ci or histo_add_pi

        if show_intervals:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
            ax2 = None

        # Histogram and KDE
        sns.histplot(self.data, kde=kde, stat="density", color="rebeccapurple", alpha=0.5, ax=ax1)
        ax1.set_ylabel("Density")
        ax1.set_xlabel(f"{name_variable}")
        ax1.set_title(f"Distribution of {name_variable}")

        if add_normal:
            y_vect = np.linspace(hat_mu - 3*hat_sigma, hat_mu + 3*hat_sigma, 100)
            ax1.plot(y_vect, norm.pdf(y_vect, hat_mu, hat_sigma), color="black", linestyle="--", label="Normal density")
            ax1.legend()

        if show_data:
            _, upper = ax1.get_ylim()
            sns.rugplot(self.data, height=0.1*upper, ax=ax1, color='black')

        # Interval annotations (confidence/prediction)
        if show_intervals:
            ax2.set_yticks([])
            ax2.set_xlabel(f"{name_variable}")
            ax2.set_ylim(0, 0.5)

            # Helper to plot a horizontal interval
            def plot_interval(ax, y_val, low, high, label, color):
                ax.hlines(y_val, low, high, color=color, linewidth=2, label=label)
                ax.scatter((low + high)/2, y_val, color=color, s=30, zorder=5)
                ax.text(high, y_val, f" {label}", va="center", fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='whitesmoke', edgecolor='gray'))

            # Confidence Intervals
            ci_y = 0.4
            if histo_add_ci:
                if histo_choose_ci in ["Mean", "Both"]:
                    plot_interval(ax2, ci_y, self.ci_mean[0], self.ci_mean[1], "CI Mean", "blue")
                if histo_choose_ci in ["Median", "Both"]:
                    plot_interval(ax2, ci_y - 0.1, self.ci_median[0], self.ci_median[1], "CI Median", "green")

            # Prediction Intervals
            pi_y = 0.1
            if histo_add_pi:
                if histo_choose_pi == "Mean":
                    plot_interval(ax2, pi_y, self.pi_mean[0], self.pi_mean[1], "PI Mean", "darkred")
                elif histo_choose_pi == "Median":
                    plot_interval(ax2, pi_y, self.pi_median[0], self.pi_median[1], "PI Median", "darkred")
                elif histo_choose_pi == "IQR":
                    plot_interval(ax2, pi_y, self.pi_iqr[0], self.pi_iqr[1], "PI IQR", "darkred")
                elif histo_choose_pi == "Bootstrap":
                    plot_interval(ax2, pi_y, self.pi_boots[0], self.pi_boots[1], "PI Bootstrap", "darkred")

        return fig
    
    # --- Plot ECDF ---
    def PlotEcdf(self, name_variable, alpha, confidence, add_normal, hat_mu, hat_sigma):

        ecdf = ECDF(self.data)
        
        plt.style.use("seaborn-v0_8-whitegrid")  # Consistent styling
        
        fig, ax = plt.subplots(figsize=(8, 5))

        # ECDF step plot
        ax.step(ecdf.x, ecdf.y, where='post', color='rebeccapurple', linewidth=2, label="ECDF")

        # Optional: scatter markers (remove if redundant)
        ax.scatter(ecdf.x, ecdf.y, color='rebeccapurple', s=10, alpha=0.6)

        # Confidence band using DKW inequality
        if confidence:
            epsilon = np.sqrt(np.log(2 / alpha) / (2 * self.n))
            lower = np.clip(ecdf.y - epsilon, 0, 1)
            upper = np.clip(ecdf.y + epsilon, 0, 1)
            ax.fill_between(ecdf.x, lower, upper, step='post', color='plum', alpha=0.4, label='DKW CI')

        # Optional: add normal CDF for comparison
        if add_normal:
            y_vals = np.linspace(hat_mu - 3 * hat_sigma, hat_mu + 3 * hat_sigma, 100)
            ax.plot(y_vals, norm.cdf(y_vals, hat_mu, hat_sigma), color='black', linestyle='--', linewidth=2, label="Normal CDF")
            ax.set_xlim(min(self.data.min(), y_vals.min()) - 0.1, max(self.data.max(), y_vals.max()) + 0.1)
        else:
            ax.set_xlim(self.data.min() - 0.1, self.data.max() + 0.1)

        # Axis labels and title
        ax.set_title("Empirical Cumulative Distribution Function", fontsize=14)
        ax.set_xlabel(f"{name_variable}", fontsize=12)
        ax.set_ylabel("ECDF", fontsize=12)
        ax.set_ylim(0, 1.05)

        # Gridlines and legend
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="lower right", fontsize=10)

        return fig

    # --- Plot Confidence Regions ---
    def PlotConfidenceRegions(self, probs, eps_mu, eps_sigma, add):

        # Reverse probabilities to get increasing chi2 levels
        probs = probs[::-1]
        levels = np.exp(-0.5 * chi2.ppf(probs, 2))

        # Grids for mu and sigma
        mu_vect = np.linspace(self.ci_mean[0] - eps_mu[0], self.ci_mean[1] + eps_mu[1], 200)
        sigma_vect = np.linspace(self.ci_deviation[0] - eps_sigma[0], self.ci_deviation[1] + eps_sigma[1], 200)
        mu_grid, sigma_grid = np.meshgrid(mu_vect, sigma_vect)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Contour plot of relative likelihood
        Z = self.RelativeLikelihood(mu_grid, sigma_grid)
        contour = ax.contour(mu_grid, sigma_grid, Z, levels=levels, cmap="plasma")

        # Mark the MLE estimate
        ax.scatter(self.mean, self.S0, color='black', s=60, label="MLE", zorder=5)

        # Optional: add rectangular CI box
        if add:
            ci_x = [self.ci_mean[0], self.ci_mean[1], self.ci_mean[1], self.ci_mean[0], self.ci_mean[0]]
            ci_y = [self.ci_deviation[0], self.ci_deviation[0], self.ci_deviation[1], self.ci_deviation[1], self.ci_deviation[0]]
            ax.plot(ci_x, ci_y, color='red', linestyle='--', label="CI box")

        # Axis and title
        ax.set_title(r"Confidence Regions for $\mu$ and $\sigma$", fontsize=14)
        ax.set_xlabel(r"$\mu$", fontsize=12)
        ax.set_ylabel(r"$\sigma$", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Format legend from contour handles
        handles, _ = contour.legend_elements()
        formatted_probs = [f"{100*p:.1f}%" for p in probs]
        ax.legend(handles + [ax.collections[-1]], formatted_probs + ["MLE"], loc="upper right", frameon=True)

        plt.tight_layout()
        return fig


# # 🧠 🌏 General functions

# In[4]:


# --- MODIFIED BLOCK ---
def download_table_as_csv(filename):
    """
    Creates a temporary CSV file from the cached DataFrame and returns its path for download.
    Accepts a user-provided filename.
    """
    df = export_cache.get("table")
    if df is None:
        gr.Warning("❌ Error: No table available to download.")
        return None
    
    # Sanitize and set a default filename
    if not filename or not filename.strip():
        base_filename = "statistical_summary"
    else:
        # Remove characters that are invalid in filenames
        base_filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        base_filename = base_filename.strip()
        if not base_filename: # If sanitization results in an empty string
            base_filename = "statistical_summary"

    try:
        # Use a temporary directory which Gradio will manage
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', prefix=base_filename, suffix='.csv', encoding='utf-8') as tmpfile:
            df.to_csv(tmpfile.name, index=False)
            return tmpfile.name
    except Exception as e:
        gr.Error(f"❌ Error creating CSV file: {e}")
        return None
    
def download_figure_as_image(filename):
    """
    Creates a temporary PNG file from the cached figure and returns its path for download.
    Accepts a user-provided filename.
    """
    fig = export_cache.get("figure")
    if fig is None:
        gr.Warning("❌ Error: No figure available to download.")
        return None

    # Sanitize and set a default filename
    if not filename or not filename.strip():
        base_filename = "output_figure"
    else:
        # Remove characters that are invalid in filenames
        base_filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        base_filename = base_filename.strip()
        if not base_filename: # If sanitization results in an empty string
            base_filename = "output_figure"

    try:
        with tempfile.NamedTemporaryFile(delete=False, prefix=base_filename, suffix='.png') as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            return tmpfile.name
    except Exception as e:
        gr.Error(f"❌ Error creating image file: {e}")
        return None


# In[5]:


def blank_plot():
    fig, ax = plt.subplots()
    ax.axis('off')
    return fig

def load_numeric_cols():
    numeric_cols = df_cache.get("numeric_cols", [])
    selected = numeric_cols[0] if numeric_cols else None
    return gr.update(choices=numeric_cols, value=selected)


# In[6]:


def prepare_data(column):
    # --- Read data and validate ---
    original_df = df_cache.get("df")
    filtered_df = df_cache.get("filtered_df")

    if original_df is None:
        return None, None, None, pd.DataFrame([["Please upload a valid CSV."]], columns=["Error"]), blank_plot()

    # --- Use filtered data if it differs from original ---
    df = filtered_df if filtered_df is not None and not filtered_df.equals(original_df) else original_df

    # --- Select numeric column ---
    if column not in df.columns:
        return None, None, None, pd.DataFrame([["Selected column is not in the dataframe."]], columns=["Error"]), blank_plot()

    data = df[column].dropna()

    # --- Initialize or reuse Statistics object ---
    stats = df_cache.get("stats")
    if stats is None or not np.array_equal(stats.data, data.to_numpy()):
        stats = Statistics(data)
        df_cache["stats"] = stats

    return df, data, stats, None, None  # df, data, stats, error_df, error_plot


# In[7]:


def choose_mu(sel, stats, text_box):
    if sel == "Sample Mean":
        hat_mu = stats.mean
    elif sel == "Sample Median":
        hat_mu = stats.median
    elif sel == "Geometric Mean":
        hat_mu = stats.geometric_mean
    elif sel == "Harmonic Mean":
        hat_mu = stats.harmonic_mean
    elif sel == "Weighted Mean":
        hat_mu = stats.weighted_mean
    elif sel == "Trimmed Mean":
        hat_mu = stats.trimmed_mean
    elif sel == "Interquartile Mean":
        hat_mu = stats.interquartile_mean
    elif sel == "Winsorized Mean":
        hat_mu = stats.winsorized_mean
    elif sel == "Other":
        hat_mu = float(text_box)
        
    return hat_mu

def choose_sigma(sel, stats, text_box):
    if sel == "Deviation (1 ddof)":
        hat_sigma = stats.S1
    elif sel == "Range (bias corrected)":
        hat_sigma = stats.R_bias_corrected
    elif sel == "IQR (bias corrected)":
        hat_sigma = stats.IQR_bias_corrected
    elif sel == "MAD (bias corrected)":
        hat_sigma = stats.MAD_bias_corrected
    elif sel == "AAD (bias corrected)":
        hat_sigma = stats.AAD_bias_corrected
    elif sel == "Other":
        hat_sigma = float(text_box)

    return hat_sigma


# # 🎮 🌏 General logic of GUI

# In[8]:


def add_normal_warning(check):
    if check:
        gr.Warning("If you haven't done it yet, run first a descriptive analysis for central tendency and dispersion.")


# In[9]:


def parse_text(input_str):
        return float(input_str) if input_str.strip() else None


# In[10]:


def toggle_add_normal(check, sel_mu, sel_sigma):
    if check:
        if sel_mu == "Other":
            if sel_sigma == "Other":
                return [
                    gr.update(visible=True), # hat_mu
                    gr.update(visible=True), # hat_mu_text
                    gr.update(visible=True), # hat_sigma
                    gr.update(visible=True)  # hat_sigma_text
                ]
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
        else:
            if sel_sigma == "Other":
                return (
                    gr.update(visible=True), 
                    gr.update(visible=False), 
                    gr.update(visible=True), 
                    gr.update(visible=True)
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )


# # 🖥️ 🌏  General GUI blocks

# In[11]:


# --- MODIFIED BLOCK ---
def build_results_block():        
    gr.Markdown("# 🎯 Results")

    with gr.Row(visible=False, elem_id="row_centered") as output_table_group:
        name_save_table = gr.Textbox(label="Filename (without extension)", placeholder="e.g. descriptive_stats")
        download_table_button = gr.Button("💾 Download Table as CSV")
        file_output_table = gr.File(label="Download link will appear here", interactive=False)

    output_table = gr.Dataframe(visible=False)

    with gr.Row(visible=False, elem_id="row_centered") as output_plot_group:
        name_save_figure = gr.Textbox(label="Filename (without extension)", placeholder="e.g. histogram")
        download_figure_button = gr.Button("🖼️ Download Figure as PNG")
        file_output_figure = gr.File(label="Download link will appear here", interactive=False)

    output_plot = gr.Plot(visible=False)

    download_table_button.click(
        fn=download_table_as_csv,
        inputs=[name_save_table],
        outputs=[file_output_table]
    )

    download_figure_button.click(
        fn=download_figure_as_image,
        inputs=[name_save_figure],
        outputs=[file_output_figure]
    )

    return output_table_group, output_table, output_plot_group, output_plot


# In[12]:


# --- MODIFIED BLOCK ---
def build_results_block_2():       
    gr.Markdown("# 🎯 Results")
    
    with gr.Row(visible=False, elem_id="row_centered") as output_table_group:
        name_save_table = gr.Textbox(label="Filename (without extension)", placeholder="e.g. regression_coeffs")
        download_table_button = gr.Button("💾 Download Coefficient Table as CSV")
        file_output_table = gr.File(label="Download link will appear here", interactive=False)

    output_table = gr.HTML(visible=False)

    with gr.Row(visible=False, elem_id="row_centered") as output_plot_group:
        name_save_figure = gr.Textbox(label="Filename (without extension)", placeholder="e.g. regression_plot")
        download_figure_button = gr.Button("🖼️ Download Figure as PNG")
        file_output_figure = gr.File(label="Download link will appear here", interactive=False)

    output_plot = gr.Plot(visible=False)

    download_table_button.click(
        fn=download_table_as_csv,
        inputs=[name_save_table],
        outputs=[file_output_table]
    )

    download_figure_button.click(
        fn=download_figure_as_image,
        inputs=[name_save_figure],
        outputs=[file_output_figure]
    )

    return output_table_group, output_table, output_plot_group, output_plot


# # 🧠 🗄️ Brain of Data Tab

# In[13]:


def get_effective_column_types(df):
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    overrides = df_cache.get("overrides", {"num_to_cat": [], "cat_to_num": []})

    numeric = [col for col in all_numeric if col not in overrides["num_to_cat"]]
    categorical = [col for col in all_categorical if col not in overrides["cat_to_num"]]

    numeric += [col for col in overrides["cat_to_num"] if col in df.columns]
    categorical += [col for col in overrides["num_to_cat"] if col in df.columns]

    return sorted(set(numeric)), sorted(set(categorical))


# In[14]:


def load_csv(file):
    try:
        df = pd.read_csv(file.name)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        df_cache["df"] = df
        df_cache["filtered_df"] = df
        df_cache["stats"] = None

        numeric_cols, categorical_cols = get_effective_column_types(df)

        df_cache["numeric_cols"] = numeric_cols
        df_cache["categorical_cols"] = categorical_cols

        return (
            gr.update(choices=categorical_cols, value=[]),            # cat_col_dropdown
            gr.update(choices=numeric_cols, value=None),              # num_override_dropdown
            gr.update(choices=categorical_cols, value=None),          # cat_override_dropdown
            gr.update(choices=[], value=[], visible=False),           # cat_val_multiselect_1
            gr.update(choices=[], value=[], visible=False),           # cat_val_multiselect_2
            gr.update(choices=[], value=[], visible=False),           # cat_val_multiselect_3
            "CSV loaded successfully."                                # status_output (Textbox!)
        )
    except Exception as e:
        return tuple([gr.update(choices=[], value=None)] * 7 + [f"Error: {e}"])


# In[15]:


def show_data_overview(check):
    if check:
        df = df_cache.get("df")

        if df is None:
            error_df = pd.DataFrame([["Please upload a valid CSV."]], columns=["Error"])
            return error_df, gr.update(visible=True), None, gr.update(visible=False)

        # --- Compute description and data types ---
        try:
            desc = df.describe().T.round(ROUND).reset_index().rename(columns={"index": "Variable"})
        except Exception as e:
            desc = pd.DataFrame([[str(e)]], columns=["Error"])

        try:
            dtypes_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index": "Variable", 0: "Type"})
        except Exception as e:
            dtypes_df = pd.DataFrame([[str(e)]], columns=["Error"])
            
        return desc, gr.update(visible=True), dtypes_df, gr.update(visible=True)
    else:
        return None, gr.update(visible=False), None, gr.update(visible=False)


# In[16]:


def show_data_overview_filter(check):
    if check:
        original_df = df_cache.get("df")
        filtered_df = df_cache.get("filtered_df")

        if original_df is None:
            error_df = pd.DataFrame([["Please upload a valid CSV."]], columns=["Error"])
            return error_df, gr.update(visible=True), None, gr.update(visible=False)

        # --- Use filtered data if it differs from original ---
        df = filtered_df if filtered_df is not None and not filtered_df.equals(original_df) else original_df

        # --- Compute description and data types ---
        try:
            desc = df.describe().T.round(ROUND).reset_index().rename(columns={"index": "Variable"})
        except Exception as e:
            desc = pd.DataFrame([[str(e)]], columns=["Error"])

        try:
            dtypes_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index": "Variable", 0: "Type"})
        except Exception as e:
            dtypes_df = pd.DataFrame([[str(e)]], columns=["Error"])
            
        return desc, gr.update(visible=True), dtypes_df, gr.update(visible=True)
    else:
        return None, gr.update(visible=False), None, gr.update(visible=False)


# In[17]:


def reclassify_as_categorical(column):
    numeric_cols = df_cache.get("numeric_cols", [])
    categorical_cols = df_cache.get("categorical_cols", [])

    if column and column in numeric_cols:
        numeric_cols.remove(column)
        categorical_cols.append(column)
        df_cache["numeric_cols"] = numeric_cols
        df_cache["categorical_cols"] = categorical_cols

        return (
            gr.update(choices=categorical_cols),                      # cat_col_dropdown
            gr.update(choices=numeric_cols),                          # num_override_dropdown
            gr.update(choices=categorical_cols),                      # cat_override_dropdown
            f"Column '{column}' reclassified as categorical."         # status
        )
    else:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            f"Column '{column}' is not currently classified as numeric."
        )


# In[18]:


def reclassify_as_numeric(column):
    numeric_cols = df_cache.get("numeric_cols", [])
    categorical_cols = df_cache.get("categorical_cols", [])

    if column and column in categorical_cols:
        categorical_cols.remove(column)
        numeric_cols.append(column)
        df_cache["categorical_cols"] = categorical_cols
        df_cache["numeric_cols"] = numeric_cols

        return (
            gr.update(choices=categorical_cols),                      # cat_col_dropdown
            gr.update(choices=numeric_cols),                          # num_override_dropdown
            gr.update(choices=categorical_cols),                      # cat_override_dropdown
            f"Column '{column}' reclassified as numeric."             # status
        )
    else:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            f"Column '{column}' is not currently classified as categorical."
        )


# In[19]:


# Only 3 category filters are supported.
def update_category_filters(selected_columns):
    df = df_cache.get("df")

    if df is None or not selected_columns:
        # Hide all category selectors if nothing is selected
        return [gr.update(visible=False, choices=[], value=[]) for _ in range(3)]

    updates = []
    for i in range(3):
        if i < len(selected_columns):
            col = selected_columns[i]
            if col in df.columns:
                values = sorted(df[col].dropna().unique().tolist())
                updates.append(gr.update(visible=True, choices=values, value=[]))
            else:
                updates.append(gr.update(visible=False, choices=[], value=[]))
        else:
            updates.append(gr.update(visible=False, choices=[], value=[]))

    return updates


# In[20]:


def apply_filters(cat_cols, val1, val2, val3):
    df = df_cache.get("df")
    if df is None:
        return "❌ No data loaded."

    filtered_df = df.copy()
    category_filters = [val1, val2, val3]

    if not cat_cols or all(not vals for vals in category_filters):
        # No filters applied
        df_cache["filtered_df"] = df
        return "⚠️ No filters selected. Using full dataset."

    for i, col in enumerate(cat_cols[:3]):
        selected_vals = category_filters[i]
        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    df_cache["filtered_df"] = filtered_df
    return f"✅ Filter applied. Rows remaining: {len(filtered_df)}"


# # 🎮🗄️ Logic control of Data Tab

# In[21]:


def max_categorical_warning(check):
    if check:
        gr.Info("The maximum number of categorical columns for filter is 3.")


# In[22]:


def toggle_preview(check):
    df = df_cache.get("df")
    if check:
        return df.head(5), gr.update(visible=True)
    else:
        return pd.DataFrame(), gr.update(visible=False) # csv_preview, csv_preview
    
def toggle_preview_filter(check):
    df = df_cache.get("filtered_df")
    if check:
        return df.head(5), gr.update(visible=True)
    else:
        return pd.DataFrame(), gr.update(visible=False) # csv_preview_filter, csv_preview_filter


# # 🖥️ 🗄️ GUI of Data Tab

# In[23]:


def build_data_tab():
    #gr.Markdown("📁 File Explorer")
    with gr.Row():
        file_input = gr.File(file_types=[".csv"], label="Upload CSV")
        status_output = gr.Textbox(label="Status", interactive=False)
        preview_checkbox = gr.Checkbox(label="Show CSV Preview", value=False)
        overview_checkbox = gr.Checkbox(label="Show Dataset Summary", value=False)

    csv_preview = gr.Dataframe(label="CSV Preview", visible=False)
    desc_output = gr.Dataframe(label="Descriptive Summary", visible=False)
    dtypes_output = gr.Dataframe(label="Variable Types", visible=False)

    with gr.Accordion(label="➖ Filter Data", open=False) as filter_accordion:       
        with gr.Row():
            cat_col_dropdown = gr.Dropdown(label="Select Categorical Columns for Filter", multiselect=True, max_choices=3, elem_classes="data_related", elem_id="custom_dropdown")
            cat_val_multiselect_1 = gr.Dropdown(label="Categories for Filter 1", multiselect=True, visible=False, interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
            cat_val_multiselect_2 = gr.Dropdown(label="Categories for Filter 2", multiselect=True, visible=False, interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
            cat_val_multiselect_3 = gr.Dropdown(label="Categories for Filter 3", multiselect=True, visible=False, interactive=True, elem_classes="data_related", elem_id="custom_dropdown")

        with gr.Row(elem_id="row_centered"):
            apply_filter_button = gr.Button("🚀 Apply Filter")
            filter_status = gr.Textbox(label="Filter Status", interactive=False)
            preview_checkbox_filter = gr.Checkbox(label="Show CSV Preview", value=False)
            overview_checkbox_filter = gr.Checkbox(label="Show Dataset Summary", value=False)
        
    csv_preview_filter = gr.Dataframe(label="CSV Preview", visible=False)
    desc_output_filter = gr.Dataframe(label="Descriptive Summary", visible=False)
    dtypes_output_filter = gr.Dataframe(label="Variable Types", visible=False)

    with gr.Accordion(label="🛠️ Fix Variable Type", open=False):
        with gr.Row(elem_id="row_centered"):
            # Reclassify numeric ➝ categorical
            num_override_dropdown = gr.Dropdown(label="Reclassify Numeric Column as Categorical", elem_classes="data_related", elem_id="custom_dropdown")
            fix_to_categorical_button = gr.Button("Reclassify as Categorical")

            # Reclassify categorical ➝ numeric
            cat_override_dropdown = gr.Dropdown(label="Reclassify Categorical Column as Numeric", elem_classes="data_related", elem_id="custom_dropdown")
            fix_to_numeric_button = gr.Button("Reclassify as Numeric")

            fix_dtype_status = gr.Textbox(label="Status", interactive=False)        

    # --- Modify behavior of components of the GUI ---
    file_input.change(
        fn=load_csv,
        inputs=[file_input],
        outputs=[
            cat_col_dropdown,
            num_override_dropdown,
            cat_override_dropdown,
            cat_val_multiselect_1,
            cat_val_multiselect_2,
            cat_val_multiselect_3,
            status_output
        ]
    )

    preview_checkbox.change(
        fn=toggle_preview,
        inputs=preview_checkbox,
        outputs=[csv_preview, csv_preview]
    )

    overview_checkbox.change(
        fn=show_data_overview,
        inputs=[overview_checkbox],
        outputs=[desc_output, desc_output, dtypes_output, dtypes_output]
    )

    fix_to_categorical_button.click(
        fn=reclassify_as_categorical,
        inputs=[num_override_dropdown],
        outputs=[
            cat_col_dropdown,
            num_override_dropdown,
            cat_override_dropdown,
            fix_dtype_status
        ]
    )

    fix_to_numeric_button.click(
        fn=reclassify_as_numeric,
        inputs=[cat_override_dropdown],
        outputs=[
            cat_col_dropdown,
            num_override_dropdown,
            cat_override_dropdown,
            fix_dtype_status
        ]
    )

    cat_col_dropdown.change(
        fn=max_categorical_warning,
        inputs=cat_col_dropdown,
        outputs=[]
    )

    cat_col_dropdown.change(
        fn=update_category_filters,
        inputs=cat_col_dropdown,
        outputs=[cat_val_multiselect_1, cat_val_multiselect_2, cat_val_multiselect_3]
    )

    apply_filter_button.click(
        fn=apply_filters,
        inputs=[
            cat_col_dropdown,
            cat_val_multiselect_1,
            cat_val_multiselect_2,
            cat_val_multiselect_3
        ],
        outputs=[filter_status]
    )

    preview_checkbox_filter.change(
        fn=toggle_preview_filter,
        inputs=preview_checkbox_filter,
        outputs=[csv_preview_filter, csv_preview_filter]
    )

    overview_checkbox_filter.change(
        fn=show_data_overview_filter,
        inputs=[overview_checkbox_filter],
        outputs=[desc_output_filter, desc_output_filter, dtypes_output_filter, dtypes_output_filter]
    )


# # 🎮 📊 Logic control of Graph Tab

# In[24]:


def histo_add_ci_warning(check):
    if check:
        gr.Warning("If you haven't done it yet, run first a statistical inference for confidence interval.")

def histo_add_pi_warning(check):
    if check:
        gr.Warning("If you haven't done it yet, run first a statistical inference for prediction interval.")


# In[25]:


def toggle_graph_stat(graph_stat):
    if graph_stat == "Histogram":
        return (
            gr.update(visible=True), # histo_add_kde
            gr.update(visible=True), # histo_add_data
            gr.update(visible=True), # histo_normal_row
            gr.update(visible=True), # histo_ci_row
            gr.update(visible=True), # histo_pi_row
            gr.update(visible=False), # ecdf_normal_row
            gr.update(visible=False), # ecdf_add_conf
            gr.update(visible=False), # ecdf_alpha
        )
    elif graph_stat == "Empirical Cumulative Distribution Function (ECDF)":
        return ( 
            gr.update(visible=False), 
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )


# In[26]:


def run_graph_stat(
        column,
        graph_stat,
        histo_add_kde_check, histo_add_data_check,
        histo_add_ci, histo_choose_ci,
        histo_add_pi, histo_choose_pi,
        histo_add_normal,
        histo_hat_mu, histo_hat_mu_text,
        histo_hat_sigma, histo_hat_sigma_text,
        ecdf_add_conf,
        ecdf_alpha,
        ecdf_add_normal,
        ecdf_hat_mu, ecdf_hat_mu_text,
        ecdf_hat_sigma, ecdf_hat_sigma_text
        ):
    
    df, data, stats, error_df, error_plot = prepare_data(column)
    if error_df:
        return gr.update(visible=False), gr.update(visible=True), error_df, gr.update(visible=False), gr.update(visible=False), error_plot

    # --- Graphical Analysis ---
    if graph_stat == "Histogram":
        hat_mu, hat_sigma = None, None

        if histo_add_normal:
            hat_mu = choose_mu(histo_hat_mu, stats, histo_hat_mu_text)
            hat_sigma = choose_sigma(histo_hat_sigma, stats, histo_hat_sigma_text)

        fig = stats.PlotHistogram(
            column,
            histo_add_kde_check,
            histo_add_data_check,
            histo_add_ci, histo_choose_ci,
            histo_add_pi, histo_choose_pi,
            histo_add_normal,
            hat_mu,
            hat_sigma
        )
        
    elif graph_stat == "Empirical Cumulative Distribution Function (ECDF)":

        alpha = parse_text(ecdf_alpha)
        alpha = 1-alpha
        if alpha is None or not (0 < alpha < 1):
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                pd.DataFrame([["Invalid alpha value."]], columns=["Error"]),
                gr.update(visible=False),
                gr.update(visible=False),
                None]

        hat_mu, hat_sigma = None, None

        if ecdf_add_normal:
            hat_mu = choose_mu(ecdf_hat_mu, stats, ecdf_hat_mu_text)
            hat_sigma = choose_sigma(ecdf_hat_sigma, stats, ecdf_hat_sigma_text)

        fig = stats.PlotEcdf(
            column,
            alpha,
            ecdf_add_conf,
            ecdf_add_normal,
            hat_mu,
            hat_sigma
        )
    
    #export_cache["table"] = None
    export_cache["figure"] = fig

    # output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot
    return gr.update(visible=False), gr.update(visible=False), pd.DataFrame(), gr.update(visible=True), gr.update(visible=True), fig 


# # 🖥️ 📊 GUI of Graph Tab

# In[27]:


def build_graphical_tab():
    gr.Markdown("# 📊 Graphical Analysis")
    
    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        column_dropdown = gr.Dropdown(label="Select Numeric Column", choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")

        graph_stat_dropdown = gr.Dropdown(
            label="Select Graph",
            choices=[
                "Histogram",
                "Empirical Cumulative Distribution Function (ECDF)"
            ],
            value="Histogram",
            interactive=True
        )

        histo_add_kde = gr.Checkbox(label="Add KDE", value=True, visible=True, interactive=True)
        histo_add_data = gr.Checkbox(label="Show data", value=False, visible=True, interactive=True)

        ecdf_add_conf = gr.Checkbox(label="Add CI for the ECDF", value=True, visible=False, interactive=True)
        ecdf_alpha = gr.Textbox(label="Confidence level (e.g. 0.95)", value=0.95, interactive=True, visible=False)

    with gr.Row() as histo_normal_row:
        histo_add_normal = gr.Checkbox(label="Add Normal Density", value=False)

        histo_hat_mu = gr.Dropdown(
            label="μ",
            choices=[
                'Sample Mean',
                'Sample Median',
                'Geometric Mean',
                'Harmonic Mean',
                'Weighted Mean',
                'Trimmed Mean',
                'Interquartile Mean',
                'Winsorized Mean',
                "Other"
            ],
            value="Sample Mean",
            interactive=True,
            visible=False
        )
        histo_hat_mu_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

        histo_hat_sigma = gr.Dropdown(
            label="σ",
            choices=[
                "Deviation (1 ddof)",
                "Range (bias corrected)",
                "IQR (bias corrected)",
                "MAD (bias corrected)",
                "AAD (bias corrected)",
                "Other"
            ],
            value="Deviation (1 ddof)",
            interactive=True,
            visible=False
        )
        histo_hat_sigma_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

    with gr.Row() as histo_ci_row:
        histo_add_ci = gr.Checkbox(label="Add Confidence Interval", value=False)

        histo_choose_ci = gr.Radio(
            label="Confidence Interval",
            choices=["Mean", "Median", "Both"],
            value="Both",
            interactive=True,
            visible=False
        )

    with gr.Row() as histo_pi_row:                
        histo_add_pi = gr.Checkbox(label="Add Prediction Interval", value=False)
        histo_choose_pi = gr.Radio(
            label="Prediction Interval",
            choices=["Mean", "Median", "IQR", "Bootstrap"],
            value="Mean",
            interactive=True,
            visible=False
        )

    with gr.Row(visible=False) as ecdf_normal_row:
        ecdf_add_normal = gr.Checkbox(label="Add Normal CDF", value=False)

        ecdf_hat_mu = gr.Dropdown(
            label="μ",
            choices=[
                'Sample Mean',
                'Sample Median',
                'Geometric Mean',
                'Harmonic Mean',
                'Weighted Mean',
                'Trimmed Mean',
                'Interquartile Mean',
                'Winsorized Mean',
                "Other"
            ],
            value="Sample Mean",
            interactive=True,
            visible=False
        )
        ecdf_hat_mu_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

        ecdf_hat_sigma = gr.Dropdown(
            label="σ",
            choices=[
                "Deviation (1 ddof)",
                "Range (bias corrected)",
                "IQR (bias corrected)",
                "MAD (bias corrected)",
                "AAD (bias corrected)",
                "Other"
            ],
            value="Deviation (1 ddof)",
            interactive=True,
            visible=False
        )
        ecdf_hat_sigma_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

    with gr.Column(elem_id="column_centered"):
        run_graph_stat_button = gr.Button(value="🚀 Run Graphical Analysis", elem_id="run_button")

    # --- Results ---
    output_table_row, output_table, output_plot_row, output_plot = build_results_block()

    # --- Modify behavior of components of the GUI ---
    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[column_dropdown]
    )

    graph_stat_dropdown.change(
        fn=toggle_graph_stat,
        inputs=[graph_stat_dropdown],
        outputs=[
            histo_add_kde,
            histo_add_data,
            histo_normal_row,
            histo_ci_row,
            histo_pi_row,
            ecdf_normal_row,
            ecdf_add_conf,
            ecdf_alpha]
    )

    histo_add_ci.change(
        fn=lambda check: gr.update(visible=check),
        inputs=[histo_add_ci],
        outputs=[histo_choose_ci]
    )

    histo_add_pi.change(
        fn=lambda check: gr.update(visible=check),
        inputs=[histo_add_pi],
        outputs=[histo_choose_pi]
    )

    histo_add_ci.change(
        fn=histo_add_ci_warning,
        inputs=[histo_add_ci],
        outputs=[]
    )

    histo_add_pi.change(
        fn=histo_add_pi_warning,
        inputs=[histo_add_pi],
        outputs=[]
    )

    histo_add_normal.change(
        fn=toggle_add_normal,
        inputs=[histo_add_normal, histo_hat_mu, histo_hat_sigma],
        outputs=[histo_hat_mu, histo_hat_mu_text, histo_hat_sigma, histo_hat_sigma_text]
    )

    histo_add_normal.change(
        fn=add_normal_warning,
        inputs=histo_add_normal,
        outputs=[]
    )

    histo_hat_mu.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=histo_hat_mu,
        outputs=histo_hat_mu_text
    )

    histo_hat_sigma.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=histo_hat_sigma,
        outputs=histo_hat_sigma_text
    )

    ecdf_add_conf.change(
        fn=lambda check: gr.update(visible=check),
        inputs=ecdf_add_conf,
        outputs=ecdf_alpha
    )

    ecdf_add_normal.change(
        fn=toggle_add_normal,
        inputs=[ecdf_add_normal, ecdf_hat_mu, ecdf_hat_sigma],
        outputs=[ecdf_hat_mu, ecdf_hat_mu_text, ecdf_hat_sigma, ecdf_hat_sigma_text]
    )

    ecdf_add_normal.change(
        fn=add_normal_warning,
        inputs=ecdf_add_normal,
        outputs=[]
    )

    ecdf_hat_mu.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=ecdf_hat_mu,
        outputs=ecdf_hat_mu_text
    )

    ecdf_hat_sigma.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=ecdf_hat_sigma,
        outputs=ecdf_hat_sigma_text
    )

    # --- Run Analysis Button ---
    run_graph_stat_button.click(
        run_graph_stat,
        inputs=[
            column_dropdown,
            graph_stat_dropdown,
            histo_add_kde, histo_add_data,
            histo_add_ci, histo_choose_ci,
            histo_add_pi, histo_choose_pi,
            histo_add_normal,
            histo_hat_mu, histo_hat_mu_text,
            histo_hat_sigma, histo_hat_sigma_text,
            ecdf_add_conf,
            ecdf_alpha,
            ecdf_add_normal,
            ecdf_hat_mu, ecdf_hat_mu_text,
            ecdf_hat_sigma, ecdf_hat_sigma_text          
        ],
        outputs=[output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot])


# # 🎮 🧮 Logic control of Descriptive Tab

# In[28]:


def parse_weights(input_str, length):
    if not input_str.strip():
        return None
    weights = [float(w.strip()) for w in input_str.split(',') if w.strip()]
    if len(weights) != length:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of observations ({length})")
    return weights

def parse_winsor(input_str):
    if not input_str.strip():
        return None
    winsor_param = [float(w.strip()) for w in input_str.split(',') if w.strip()]
    if len(winsor_param) != 2:
        raise ValueError(f"Length of winsorized input ({len(winsor_param)}) must be two (lower, and upper)")
    return winsor_param

def parse_quantiles(input_str):
    if ',' in input_str:
        q = [float(x.strip()) for x in input_str.split(',') if x.strip()]
    else:
        q = float(input_str)
    return q


# In[29]:


def toggle_desc_params(desc_stat):
    if desc_stat == "Quantiles":
        return (
            gr.update(visible=True),  # quantiles input
            gr.update(visible=False) # central_tendecy_params
        )
    elif desc_stat in ["Central Tendency", "All Descriptive Statistics"]:
        return (
            gr.update(visible=False),  
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=False),  
            gr.update(visible=False)
        )


# In[30]:


def run_desc_stat(
        column,
        descriptive_stat,
        quantiles_input, weights_input, trim_input, winsor_input
        ):
    
    df, data, stats, error_df, error_plot = prepare_data(column)
    if error_df:
        return gr.update(visible=False), gr.update(visible=True), error_df, gr.update(visible=False), gr.update(visible=False), error_plot 

    # --- Descriptive Analysis ---
    if descriptive_stat == "Quantiles":
        q = parse_quantiles(quantiles_input)
        stats.CalculateQuantiles(q)
        df_output = stats.quantiles.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "Quartiles":
        stats.CalculateQuartiles()
        df_output = stats.quartiles.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "Central Tendency":
        trim_param = parse_text(trim_input)
        winsor_param = parse_winsor(winsor_input)
        weights = parse_weights(weights_input, len(data))

        stats.CalculateCentralTendency(weights=weights, winsor_param=winsor_param, trim_param=trim_param)
        df_output = stats.central_tendency.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "Dispersion":
        stats.CalculateDispersion()
        df_output = stats.dispersion.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "Skewness":
        stats.CalculateSkewness()
        df_output = stats.skew.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "Kurtosis":
        stats.CalculateKurtosis()
        df_output = stats.kurtosis.round(ROUND).reset_index().rename(columns={"index": "Measure"})

    elif descriptive_stat == "All Descriptive Statistics":
        trim_param = parse_text(trim_input)
        winsor_param = parse_winsor(winsor_input)
        weights = parse_weights(weights_input, len(data))

        stats.CalculateDescriptiveStatistics(weights=weights, winsor_param=winsor_param, trim_param=trim_param)

        # Merge all tables with a hierarchical index
        df_combined = pd.concat([
            stats.quartiles,
            stats.central_tendency,
            stats.dispersion,
            stats.skew,
            stats.kurtosis
        ], keys=["Quartiles", "Central Tendency", "Dispersion", "Skewness", "Kurtosis"])

        df_output = df_combined.round(ROUND).reset_index().rename(columns={"level_0": "Statistic Type", "level_1": "Measure"})

    export_cache["table"] = df_output
    #export_cache["figure"] = fig

    # output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot
    return gr.update(visible=True), gr.update(visible=True), df_output, gr.update(visible=False), gr.update(visible=False), None 


# # 🖥️ 🧮 GUI of Descriptive Tab

# In[31]:


def build_descriptive_tab():
    gr.Markdown("# 🧮 Descriptive Analysis")

    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        column_dropdown = gr.Dropdown(label="Select Numeric Column", choices=[], elem_classes="data_related", elem_id="custom_dropdown")

        descriptive_stat = gr.Dropdown(
            label="Select Descriptive Statistic",
            choices=[
                "Quantiles",
                "Quartiles",
                "Central Tendency",
                "Dispersion",
                "Skewness",
                "Kurtosis",
                "All Descriptive Statistics"
            ],
            value="All Descriptive Statistics",
            interactive=True
        )
        quantiles_input = gr.Textbox(label="Quantiles (e.g., 0.25, 0.5, 0.75)", value="0.25, 0.5, 0.75", visible=False)

    with gr.Row() as central_tendecy_params:
        weights_input = gr.Textbox(label="Weights (comma-separated)", placeholder="e.g., 1, 1, 0.5, 0.8", visible=True)
        trim_input = gr.Textbox(label="Trim percentage (e.g., 0.1)", value=0.1, visible=True)
        winsor_input = gr.Textbox(label="Winsorized percentages (e.g., 0.1, 0.1)", value="0.1, 0.1", visible=True)

    with gr.Column(elem_id="column_centered"):
        run_desc_stat_button = gr.Button(value="🚀 Run Descriptive Analysis", elem_id="run_button")

    # --- Results ---
    output_table_row, output_table, output_plot_row, output_plot = build_results_block()

    # --- Modify behavior of components of the GUI ---
    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[column_dropdown]
    )

    descriptive_stat.change(
        fn=toggle_desc_params,
        inputs=descriptive_stat,
        outputs=[quantiles_input, central_tendecy_params]
    )

    # --- Run Analysis Button ---
    run_desc_stat_button.click(
        run_desc_stat,
        inputs=[
            column_dropdown,
            descriptive_stat,
            quantiles_input, weights_input, trim_input, winsor_input
        ],
        outputs=[output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot]
    )


# # 🎮 💭 Logic control of Statistical Inference Tab

# In[32]:


def stat_inf_warning(check):
    if check:
        gr.Warning("If you haven't done it yet, run first a descriptive analysis for central tendency and dispersion.")

def conf_interval_warning(check):
    if check == "Confidence Regions":
        gr.Warning("If you haven't done it yet, run first a statistical inference for CI.")

def conf_interval_range(check):
    if check == "Range (bias corrected)":
        gr.Info("CI calculations for deviation can be slow for n larger than 25, due to the complexity of the integrals.")


# In[33]:


def parse_probs(input_str):
    if not input_str.strip():
        return None
    probs = [float(w.strip()) for w in input_str.split(',') if w.strip()]
    return probs

def parse_margin(input_str):
    if not input_str.strip():
        return None
    eps = [float(w.strip()) for w in input_str.split(',') if w.strip()]
    if len(eps) != 2:
        raise ValueError(f"Length of margin ({len(eps)}) must be two (μ,σ)")
    return eps


# In[34]:


def toggle_stat_inf(sel, boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check):
    if sel == "Confidence Interval":
        return [
            gr.update(visible=True), # alpha_input
            gr.update(visible=True), # stat_inf_intervals
            gr.update(visible=True, value=boots_mean_check), # boots_mean_check
            gr.update(visible=True, value=boots_median_check), # boots_median_check
            gr.update(visible=True, value=boots_deviation_check), # boots_deviation_check
            gr.update(visible=False, value=False), # boots_prediction_check
            gr.update(visible=False) # stat_inf_regions
        ]
    elif sel == "Prediction Interval":
        if boots_prediction_check:
            return [
                gr.update(visible=True), # alpha_input
                gr.update(visible=False), # stat_inf_intervals
                gr.update(visible=False, value=False), # boots_mean_check
                gr.update(visible=False, value=False), # boots_median_check
                gr.update(visible=False, value=False), # boots_deviation_check
                gr.update(visible=True, value=boots_prediction_check), # boots_prediction_check
                gr.update(visible=False) # stat_inf_regions
            ]
        else:
            return [
                gr.update(visible=True), # alpha_input
                gr.update(visible=True), # stat_inf_intervals
                gr.update(visible=False, value=False), # boots_mean_check
                gr.update(visible=False, value=False), # boots_median_check
                gr.update(visible=False, value=False), # boots_deviation_check
                gr.update(visible=True, value=boots_prediction_check), # boots_prediction_check
                gr.update(visible=False) # stat_inf_regions
            ]
    elif sel == "Confidence and Prediction Intervals":
            return [
                gr.update(visible=True), # alpha_input
                gr.update(visible=True), # stat_inf_intervals
                gr.update(visible=True, value=boots_mean_check), # boots_mean_check
                gr.update(visible=True, value=boots_median_check), # boots_median_check
                gr.update(visible=True, value=boots_deviation_check), # boots_deviation_check
                gr.update(visible=True, value=boots_prediction_check), # boots_prediction_check
                gr.update(visible=False) # stat_inf_regions
            ]
    elif sel == "Confidence Regions":
            return [
                gr.update(visible=False), # alpha_input
                gr.update(visible=False), # stat_inf_intervals
                gr.update(visible=False, value=False), # boots_mean_check
                gr.update(visible=False, value=False), # boots_median_check
                gr.update(visible=False, value=False), # boots_deviation_check
                gr.update(visible=False, value=False), # boots_prediction_check
                gr.update(visible=True) # stat_inf_regions
            ]
    elif sel == "Confidence Interval and Regions":
            return [
                gr.update(visible=True), # alpha_input
                gr.update(visible=True), # stat_inf_intervals
                gr.update(visible=True, value=boots_mean_check), # boots_mean_check
                gr.update(visible=True, value=boots_median_check), # boots_median_check
                gr.update(visible=True, value=boots_deviation_check), # boots_deviation_check
                gr.update(visible=False, value=False), # boots_prediction_check
                gr.update(visible=True) # stat_inf_regions
            ]
    
def toggle_slider_visibility(boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check):
    # Show slider if any checkbox is checked
    visible = boots_mean_check or boots_median_check or boots_deviation_check or boots_prediction_check
    return gr.update(visible=visible)


# In[35]:


def update_mean_choices(check):
    if check:
        return gr.update(
            choices=[
                "Sample Mean",
                "Geometric Mean",
                "Harmonic Mean",
                "Trimmed Mean",
                "Interquartile Mean",
                "Winsorized Mean"
            ],
            value="Sample Mean")
    else:
        return gr.update(
            choices=[
                "Sample Mean",
                "Geometric Mean",
                "Harmonic Mean",
                "Weighted Mean",
                "Trimmed Mean",
                "Interquartile Mean",
                "Winsorized Mean",
                "Other"
            ],
            value="Sample Mean")


# In[36]:


def update_median_choices(check):
    if check:
        return gr.update(
            choices=[
                "Sample Median"
            ],
            value="Sample Median")
    else:
        return gr.update(
            choices=[
                "Sample Median",
                "Other"
            ],
            value="Sample Median")


# In[37]:


def update_deviation_choices(check):
    if check:
        return gr.update(
            choices=[
                "Deviation (1 ddof)",
                "Range (bias corrected)",
                "IQR (bias corrected)",
                "MAD (bias corrected)",
                "AAD (bias corrected)"
            ],
            value="Deviation (1 ddof)")
    else:
        return gr.update(
            choices=[
                "Deviation (1 ddof)",
                "Range (bias corrected)",
                "IQR (bias corrected)",
                "MAD (bias corrected)",
                "AAD (bias corrected)",
                "Other"
            ],
            value="Deviation (1 ddof)")


# In[38]:


def run_stat_inf(
        column,
        statistical_inf,
        alpha_input,
        probs_input, eps_input_mu, eps_input_sigma,
        like_add_interval,
        mean_select, mean_estimate_text,
        median_select, median_estimate_text,
        sigma_select, sigma_estimate_text,
        boots_mean, boots_median, boots_deviation, boots_prediction,
        bootstrap_samples
        ):
    
    df, data, stats, error_df, error_plot = prepare_data(column)
    if error_df:
        return gr.update(visible=False), gr.update(visible=True), error_df, gr.update(visible=False), gr.update(visible=False), error_plot 

    # --- Statistical Inference ---
    alpha = parse_text(alpha_input)
    alpha = 1 - alpha
    if alpha is None or not (0 < alpha < 1):
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            pd.DataFrame([["Invalid alpha value."]], columns=["Error"]),
            gr.update(visible=False),
            gr.update(visible=False),
            None]

    # Choose mean
    hat_mean = choose_mu(mean_select, stats, mean_estimate_text)
    
    # Choose median
    if median_select == "Sample Median":
        hat_median = stats.median
    elif  median_select == "Other":
        hat_median = float(median_estimate_text)

    # Choose sigma
    hat_sigma = choose_sigma(sigma_select, stats, sigma_estimate_text)
        
    if (mean_select == "Sample Mean") and (sigma_select == "Deviation (1 ddof)"):
        dist="t"
    else:
        dist="norm"

    if statistical_inf == "Confidence Interval":
        stats.CalculateConfidenceInterval(
            alpha, hat_mean, hat_median, hat_sigma, dist,
            mean_select, sigma_select, boots_mean, boots_median, boots_deviation, bootstrap_samples)
        df_output = stats.confidence_intervals.round(ROUND).reset_index().rename(columns={"index": "Measure"})
        fig = None
        visible_table = True
        visible_fig = False

    elif statistical_inf == "Prediction Interval":
        if boots_prediction:
            stats.CalculatePiBoots(alpha, bootstrap_samples)
            df_output = stats.prediction_intervals_boots.round(ROUND).reset_index().rename(columns={"index": "Measure"})
        else:
            stats.CalculatePredictionInterval(alpha, hat_mean, hat_median, hat_sigma, dist)
            df_output = stats.prediction_intervals.round(ROUND).reset_index().rename(columns={"index": "Measure"})
        fig = None
        visible_table = True
        visible_fig = False

    elif statistical_inf == "Confidence and Prediction Intervals":
        stats.CalculateConfidenceInterval(
            alpha, hat_mean, hat_median, hat_sigma, dist,
            mean_select, sigma_select, boots_mean, boots_median, boots_deviation, bootstrap_samples)
        if boots_prediction:
            stats.CalculatePiBoots(alpha, bootstrap_samples)

            # Merge all tables with a hierarchical index
            df_combined = pd.concat([
                stats.confidence_intervals,
                stats.prediction_intervals_boots
            ], keys=["Confidence", "Prediction"])
        else:
            stats.CalculatePredictionInterval(alpha, hat_mean, hat_median, hat_sigma, dist)

            # Merge all tables with a hierarchical index
            df_combined = pd.concat([
                stats.confidence_intervals,
                stats.prediction_intervals
            ], keys=["Confidence", "Prediction"])
    
        df_output = df_combined.round(ROUND).reset_index().rename(columns={"level_0": "Interval Type", "level_1": "Measure"})
        fig = None
        visible_table = True
        visible_fig = False

    elif statistical_inf == "Confidence Regions":
        probs = parse_probs(probs_input)
        eps_mu = parse_margin(eps_input_mu)
        eps_sigma = parse_margin(eps_input_sigma)
        fig = stats.PlotConfidenceRegions(probs, eps_mu, eps_sigma, like_add_interval)
        df_output = pd.DataFrame()
        visible_table = False 
        visible_fig = True
    
    elif statistical_inf == "Confidence Interval and Regions":
        probs = parse_probs(probs_input)
        eps_mu = parse_margin(eps_input_mu)
        eps_sigma = parse_margin(eps_input_sigma)
        stats.CalculateConfidenceInterval(
            alpha, hat_mean, hat_median, hat_sigma, dist,
            mean_select, sigma_select, boots_mean, boots_median, boots_deviation, bootstrap_samples)
        df_output = stats.confidence_intervals.round(ROUND).reset_index().rename(columns={"index": "Measure"})
        fig = stats.PlotConfidenceRegions(probs, eps_mu, eps_sigma, like_add_interval)
        visible_table = True
        visible_fig = True

    export_cache["table"] = df_output
    export_cache["figure"] = fig

    # output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot
    return gr.update(visible=visible_table), gr.update(visible=visible_table), df_output, gr.update(visible=visible_fig), gr.update(visible=visible_fig), fig


# # 🖥️ 💭 GUI of Statistical Inference Tab

# In[39]:


def build_inference_tab():
    gr.Markdown("# 💭 Statistical Inference")

    with gr.Accordion("🧠 Technical Information", open=False):
        gr.Markdown(
            """
            - All intervals are calculated assuming the observations are i.i.d. from a Normal distribution.  
            - If the sample mean and the sample deviation with one degree of freedom are selected as estimators for the mean and standard deviation, 
            then a *t*-distribution is used to compute the Confidence Interval (CI) for the mean and the Prediction Interval (PI) based on the mean.  
            - The asymptotic Normal distribution is used for the CI and PI based on the median.
            - The exact distribution for the CI of the deviation is used when the deviation with 1 ddof or the range is selected.
            - The asymptotic Normal distribution for the CI for the deviation is used when the IQR, AAD or MAD is selected.
            - When the sample size is not large, it might be preferable to use intervals based on bootstrap.
            """
        )

    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        column_dropdown = gr.Dropdown(label="Select Numeric Column", choices=[], elem_classes="data_related", elem_id="custom_dropdown")
        
        stat_inf_dropdown = gr.Dropdown(
            label="Type of Estimation",
            choices=[
                "Confidence Interval",
                "Prediction Interval",
                "Confidence and Prediction Intervals",
                "Confidence Regions",
                "Confidence Interval and Regions"
            ],
            value="Confidence and Prediction Intervals",
            interactive=True
        )
        alpha_input = gr.Textbox(label="Confidence level (e.g. 0.95)", value=0.95, interactive=True)

    with gr.Row(visible=True) as stat_inf_intervals:
        mean_select = gr.Dropdown(
            choices=[
                "Sample Mean",
                "Geometric Mean",
                "Harmonic Mean",
                "Weighted Mean",
                "Trimmed Mean",
                "Interquartile Mean",
                "Winsorized Mean",
                "Other"
                ],
            label="Mean Estimate",
            value="Sample Mean"
        )

        mean_estimate_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

        median_select = gr.Dropdown(
            choices=["Sample Median", "Other"],
            label="Median Estimate", value="Sample Median"
        )
        median_estimate_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

        sigma_select = gr.Dropdown(
            choices=[
                "Deviation (1 ddof)",
                "Range (bias corrected)",
                "IQR (bias corrected)",
                "MAD (bias corrected)",
                "AAD (bias corrected)",
                "Other"
            ],
            label="Deviation Estimate",
            value="Deviation (1 ddof)"
        )
        sigma_estimate_text = gr.Textbox(label="Write the value of a consistent estimator", visible=False)

    with gr.Row(visible=True) as stat_inf_boots:
        boots_mean_check = gr.Checkbox(label="Use bootstrap for the mean?", value=False)
        boots_median_check = gr.Checkbox(label="Use bootstrap for the median?", value=False)
        boots_deviation_check = gr.Checkbox(label="Use bootstrap for the deviation?", value=False)
        boots_prediction_check = gr.Checkbox(label="Use bootstrap for the prediction?", value=False)
    
    boots_sample = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="Bootstrap Samples", visible=False)

    with gr.Row(visible=False) as stat_inf_regions:
        like_probs = gr.Textbox(label="Confidence levels (from lower to higher)", value="0.1, 0.5, 0.75, 0.89, 0.95", interactive=True, visible=True)
        like_eps_mu = gr.Textbox(label="Extra margin for μ and σ", value="0.1, 0.1", interactive=True, visible=True)
        like_eps_sigma = gr.Textbox(label="Extra margin for σ", value="0.05, 0.05", interactive=True, visible=True)
        like_add_interval = gr.Checkbox(label="Add CI for μ and σ", value=True)

    with gr.Column(elem_id="column_centered"):
        run_stat_inf_button = gr.Button(value="🚀 Run Statistical Inference", elem_id="run_button")
    
    # --- Results ---
    output_table_row, output_table, output_plot_row, output_plot = build_results_block()

    # --- Modify behavior of components of the GUI ---
    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[column_dropdown]
    )

    column_dropdown.change(
        fn=stat_inf_warning,
        inputs=[column_dropdown],
        outputs=[]
    )

    stat_inf_dropdown.change(
        fn=conf_interval_warning,
        inputs=[stat_inf_dropdown],
        outputs=[]
    )

    stat_inf_dropdown.change(
        fn=toggle_stat_inf,
        inputs=[stat_inf_dropdown, boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check],
        outputs=[alpha_input, stat_inf_intervals, boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check, stat_inf_regions]
    )

    boots_prediction_check.change(
        fn=toggle_stat_inf,
        inputs=[stat_inf_dropdown, boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check],
        outputs=[alpha_input, stat_inf_intervals, boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check, stat_inf_regions]
    )

    mean_select.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=mean_select,
        outputs=mean_estimate_text
    )

    median_select.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=median_select,
        outputs=median_estimate_text
    )

    sigma_select.change(
        fn=lambda choice: gr.update(visible=(choice == "Other")),
        inputs=sigma_select,
        outputs=sigma_estimate_text
    )

    sigma_select.change(
        fn=conf_interval_range,
        inputs=sigma_select,
        outputs=[]
    )
    for checkbox in [boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check]:
        checkbox.change(
            fn=toggle_slider_visibility,
            inputs=[boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check],
            outputs=boots_sample
        )

    boots_mean_check.change(
        fn=update_mean_choices,
        inputs=boots_mean_check,
        outputs=mean_select
    )

    boots_median_check.change(
        fn=update_median_choices,
        inputs=boots_median_check,
        outputs=median_select
    )

    boots_deviation_check.change(
        fn=update_deviation_choices,
        inputs=boots_deviation_check,
        outputs=sigma_select
    )

    # --- Run Analysis Button ---
    run_stat_inf_button.click(
        run_stat_inf,
        inputs=[
            column_dropdown,
            stat_inf_dropdown,
            alpha_input,
            like_probs, like_eps_mu, like_eps_sigma,
            like_add_interval,
            mean_select, mean_estimate_text,
            median_select, median_estimate_text,
            sigma_select, sigma_estimate_text,
            boots_mean_check, boots_median_check, boots_deviation_check, boots_prediction_check,
            boots_sample
        ],
        outputs=[output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot]
    )


# # 🧠 🧪 Brain of Hypothesis Testing

# In[40]:


def plot_ttest_mean_distribution(numeric_col, sample, mu0, df_output, alternative, bootstrap_samples):
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- t-test using Pingouin ---
    t_val = df_output["T"].values[0]
    p_val = df_output["p-val"].values[0]
    df = df_output["dof"].values[0]

    # --- Sample stats ---
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    se = sample_std / np.sqrt(n)

    # --- Theoretical t-distribution under H₀ ---
    x = np.linspace(mu0 - 5 * se, mu0 + 5 * se, 1000)
    t_density = t.pdf((x - mu0) / se, df) / se

    # --- Bootstrap sampling distribution of the mean (off-screen figure) ---
    boot_means = np.array([
        np.mean(np.random.choice(sample, size=n, replace=True))
        for _ in range(bootstrap_samples)
    ])

    fig_tmp, ax_tmp = plt.subplots()
    sns.kdeplot(boot_means, bw_adjust=1.2, ax=ax_tmp)
    x_kde, y_kde = ax_tmp.lines[0].get_data()
    plt.close(fig_tmp)  # Close temporary plot

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bootstrap KDE
    ax.plot(x_kde, y_kde, color="rebeccapurple", label="Bootstrap sampling dist.", linewidth=2)

    # Plot theoretical t-distribution under H₀
    ax.plot(x, t_density, color="gray", linestyle="--", linewidth=2, label=fr"$t$-distribution ($H_0$)")

    # --- Shade p-value region under t-distribution ---
    if alternative == "two-sided":
        delta = abs(sample_mean - mu0)
        lower = mu0 - delta
        upper = mu0 + delta
        mask = (x <= lower) | (x >= upper)
    elif alternative == "greater":
        mask = x >= sample_mean
    elif alternative == "less":
        mask = x <= sample_mean
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    ax.fill_between(x, 0, t_density, where=mask, color="red", alpha=0.3,
                    label=f"p-value ≈ {p_val:.3f}")

    # --- Reference lines ---
    ax.axvline(mu0, color="tab:orange", linestyle="--", linewidth=2, label=fr"$\mu_0 = {mu0}$")
    ax.axvline(sample_mean, color="black", linestyle="-", linewidth=1.5,
               label=fr"Sample mean = {sample_mean:.2f}")

    # --- Formatting ---
    ax.set_title(f"Sampling Distribution of the Mean ({numeric_col})", fontsize=14)
    ax.set_xlabel("Sample Mean", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    return fig


# In[41]:


def one_sample_ttest(numeric_col, mu0_text, alternative, graph_check, bootstrap_samples):

    df, data, stats, error_df, error_plot = prepare_data(numeric_col)
    if error_df:
        return gr.update(visible=False), gr.update(visible=True), error_df, gr.update(visible=False), gr.update(visible=False), error_plot 

    mu0 = float(mu0_text)

    try:
        sample = df[numeric_col].dropna()
        if sample.empty:
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                pd.DataFrame([["No valid data in the selected column."]], columns=["Error"]),
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

        # --- One-sample t-test ---
        df_output = pg.ttest(sample, mu0, alternative=alternative, paired=False).round(4)

        # --- Plot ---
        if graph_check:
            fig = plot_ttest_mean_distribution(numeric_col, sample, mu0, df_output, alternative, bootstrap_samples)

            export_cache["table"] = df_output
            export_cache["figure"] = fig

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=True),
                gr.update(visible=True),
                fig
            ]
        
        else:
            export_cache["table"] = df_output
            export_cache["figure"] = None

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

    except Exception as e:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            pd.DataFrame([[f"❌ Error: {e}"]], columns=["Error"]),
            gr.update(visible=False),
            gr.update(visible=False),
            None
        ]


# In[42]:


def mirror_plot(numeric_col, group1, name_group1, group2, name_group2, df_output):
    # Extract test results from df_output
    t_val = df_output["T"].values[0]
    p_val = df_output["p-val"].values[0]

    # Means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Shared binning and KDE range
    combined = np.concatenate([group1, group2])
    x_min, x_max = min(combined), max(combined)
    bin_range = np.linspace(x_min, x_max, 30)
    bin_centers = (bin_range[:-1] + bin_range[1:]) / 2
    bin_width = np.diff(bin_range)[0]
    x_vals = np.linspace(x_min, x_max, 200)

    # --- Group 1 (top) ---
    sns.histplot(group1, bins=bin_range, stat="density", kde=False,
                 color="rebeccapurple", label=name_group1, alpha=0.6, ax=ax)
    kde1 = gaussian_kde(group1)
    ax.plot(x_vals, kde1(x_vals), color="rebeccapurple", linewidth=2)
    ax.axvline(mean1, color="rebeccapurple", linestyle="--", linewidth=2,
               label=f"{name_group1} mean = {mean1:.2f}")

    # --- Group 2 (bottom, mirrored) ---
    heights2, _ = np.histogram(group2, bins=bin_range, density=True)
    ax.bar(bin_centers, -heights2, width=bin_width,
           color="tab:orange", edgecolor="black", alpha=0.6, label=name_group2)
    kde2 = gaussian_kde(group2)
    ax.plot(x_vals, -kde2(x_vals), color="tab:orange", linewidth=2)
    ax.axvline(mean2, color="tab:orange", linestyle="--", linewidth=2,
               label=f"{name_group2} mean = {mean2:.2f}")

    # Baseline
    ax.axhline(0, color="black", linewidth=1)

    # Title, labels
    ax.set_title("Mirror Plot: Two-Sample Distribution Comparison", fontsize=14)
    ax.set_xlabel(numeric_col)
    ax.set_ylabel("Density (Top ↑ vs. Bottom ↓)", fontsize=11)

    # Annotate t-test result
    ax.text(0.01, 0.95,
            f"p = {p_val:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    ax.legend()
    plt.tight_layout()
    return fig


# In[43]:


def plot_mean_distribution(group1, name_group1, group2, name_group2, bootstrap_samples, df_output):
        # Extract test results from df_output
        t_val = df_output["T"].values[0]
        p_val = df_output["p-val"].values[0]

        # Means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # Bootstrap variances
        boot1 = [np.mean(np.random.choice(group1, size=len(group1), replace=True)) for _ in range(bootstrap_samples)]
        boot2 = [np.mean(np.random.choice(group2, size=len(group2), replace=True)) for _ in range(bootstrap_samples)]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(boot1, label=f"{name_group1} mean", fill=True, color="rebeccapurple", alpha=0.6, ax=ax)
        sns.kdeplot(boot2, label=f"{name_group2} mean", fill=True, color="tab:orange", alpha=0.6, ax=ax)

        ax.axvline(mean1, color="rebeccapurple", linestyle="--", linewidth=2)
        ax.axvline(mean2, color="tab:orange", linestyle="--", linewidth=2)

        ax.set_title(f"Bootstrap Mean Distributions", fontsize=14)
        ax.set_xlabel("Mean", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Annotate test result
        ax.text(0.98, 0.95,
                f"p = {round(p_val, 3)}\n"
                f"Mean({name_group1}) = {round(mean1, 2)}\n"
                f"Mean({name_group2}) = {round(mean2, 2)}",
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                fontsize=11)

        ax.legend()
        plt.tight_layout()

        return fig


# In[44]:


def two_sample_ttest(
        numeric_col, alternative,
        cat_col1, cat_vals1, name_group1,
        cat_col2, cat_vals2, name_group2,
        graph_check, plot_type, bootstrap_samples,
        correction):

    df, data, stats, error_df, error_plot = prepare_data(numeric_col)
    if error_df:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            error_df,
            gr.update(visible=False),
            gr.update(visible=False),
            error_plot
        ] 

    try:
        # Ensure category values match the type of the actual column
        cat_vals1 = pd.Series(cat_vals1).astype(df[cat_col1].dtype)
        cat_vals2 = pd.Series(cat_vals2).astype(df[cat_col2].dtype)

        group1 = df[df[cat_col1].isin(cat_vals1)][numeric_col].dropna()
        group2 = df[df[cat_col2].isin(cat_vals2)][numeric_col].dropna()

        if group1.empty or group2.empty:
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                pd.DataFrame([["One or both groups are empty after filtering."]], columns=["Error"]),
                gr.update(visible=False),
                gr.update(visible=False),
                error_plot
            ]

        # --- t-test ---
        df_output = pg.ttest(group1, group2, alternative=alternative, paired=False, correction=correction).round(ROUND)

        # --- Plot ---
        if graph_check:
            if plot_type == "Sample Histogram":
                fig = mirror_plot(numeric_col, group1, name_group1, group2, name_group2, df_output)
            elif plot_type == "Mean Density":
                fig = plot_mean_distribution(group1, name_group1, group2, name_group2, bootstrap_samples, df_output)

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=True),
                gr.update(visible=True),
                fig
            ]
        
        else:
            export_cache["table"] = df_output
            export_cache["figure"] = None

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

    except Exception as e:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            pd.DataFrame([[f"❌ Error: {e}"]], columns=["Error"]),
            gr.update(visible=False),
            gr.update(visible=False),
            None
        ]


# In[45]:


def plot_variance_distribution(p, group1, name_group1, var1, group2, name_group2, var2, method, bootstrap_samples):
        # Bootstrap variances
        boot1 = [np.var(np.random.choice(group1, size=len(group1), replace=True), ddof=1)
                for _ in range(bootstrap_samples)]
        boot2 = [np.var(np.random.choice(group2, size=len(group2), replace=True), ddof=1)
                for _ in range(bootstrap_samples)]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(boot1, label=f"{name_group1} variance", fill=True, color="rebeccapurple", alpha=0.6, ax=ax)
        sns.kdeplot(boot2, label=f"{name_group2} variance", fill=True, color="tab:orange", alpha=0.6, ax=ax)

        ax.axvline(var1, color="rebeccapurple", linestyle="--", linewidth=2)
        ax.axvline(var2, color="tab:orange", linestyle="--", linewidth=2)

        ax.set_title(f"Bootstrap Variance Distributions\n{method}", fontsize=14)
        ax.set_xlabel("Variance", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Annotate test result
        ax.text(0.98, 0.95,
                f"{method}\n"
                f"p = {round(p, 3)}\n"
                f"Var({name_group1}) = {round(var1, 2)}\n"
                f"Var({name_group2}) = {round(var2, 2)}",
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                fontsize=11)

        ax.legend()
        plt.tight_layout()

        return fig


# In[46]:


def variance_test(numeric_col, cat_col1, cat_vals1, name_group1,
                  cat_col2, cat_vals2, name_group2,
                  test_type, graph_check, bootstrap_samples):

    df, data, stats, error_df, error_plot = prepare_data(numeric_col)
    if error_df:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            error_df,
            gr.update(visible=False),
            gr.update(visible=False),
            error_plot
        ]

    try:
        # Ensure category values match the type of the actual column
        cat_vals1 = pd.Series(cat_vals1).astype(df[cat_col1].dtype)
        cat_vals2 = pd.Series(cat_vals2).astype(df[cat_col2].dtype)

        group1 = df[df[cat_col1].isin(cat_vals1)][numeric_col].dropna()
        group2 = df[df[cat_col2].isin(cat_vals2)][numeric_col].dropna()

        if group1.empty or group2.empty:
            err = pd.DataFrame([["One or both groups are empty after filtering."]], columns=["Error"])
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                err,
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

        # --- Apply variance test ---
        if test_type == "Bartlett":
            stat, p = bartlett(group1, group2)
            method = "Bartlett's test"
        elif test_type == "Levene":
            stat, p = levene(group1, group2, center="mean")
            method = "Levene's test"
        else:
            err = pd.DataFrame([["Invalid test type selected."]], columns=["Error"])
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                err,
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

        # --- Observed variances ---
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)

        # --- Create result table ---
        df_output = pd.DataFrame({
            "Test": [method],
            "Statistic": [round(stat, ROUND)],
            "p-value": [round(p, ROUND)],
            f"Var({name_group1})": [round(var1, ROUND)],
            f"Var({name_group2})": [round(var2, ROUND)]
        })

        # --- Create plot ---
        if graph_check:
            fig = plot_variance_distribution(p, group1, name_group1, var1, group2, name_group2, var2, method, bootstrap_samples)

            export_cache["table"] = df_output
            export_cache["figure"] = fig

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=True),
                gr.update(visible=True),
                fig
            ]
        
        else:
            export_cache["table"] = df_output
            export_cache["figure"] = None

            return [
                gr.update(visible=True),
                gr.update(visible=True),
                df_output,
                gr.update(visible=False),
                gr.update(visible=False),
                None
            ]

    except Exception as e:
        err = pd.DataFrame([[f"❌ Error: {e}"]], columns=["Error"])
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            err,
            gr.update(visible=False),
            gr.update(visible=False),
            None
        ]


# In[47]:


def one_way_anova_plot(data_group, numeric_col, cat_col, df_output):

    f_val = df_output["F"].values[0]
    p_val = df_output["p-unc"].values[0]

    # Unique groups and color palette
    groups = sorted(data_group[cat_col].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=len(groups))
    group_color_map = {group: color for group, color in zip(groups, palette)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot KDEs manually, one per group
    for group in groups:
        subset = data_group[data_group[cat_col] == group][numeric_col].dropna()
        sns.kdeplot(
            subset,
            fill=True,
            common_norm=False,
            color=group_color_map[group],
            alpha=0.5,
            linewidth=1,
            label=str(group),
            ax=ax
        )

    # Overall mean
    overall_mean = data_group[numeric_col].mean()
    ax.axvline(overall_mean, color="black", linestyle=":", linewidth=1.2, label="Overall mean")

    ax.legend(title=cat_col)

    # Add group means with matching colors
    group_means = data_group.groupby(cat_col)[numeric_col].mean()
    for group, mean_val in group_means.items():
        ax.axvline(mean_val, color=group_color_map[group], linestyle="--", linewidth=1.5, label=f"{group} mean")

    # Annotation with F and p
    ax.text(0.98, 0.95,
            f"p = {p_val:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            fontsize=11)

    # Labels and title
    ax.set_title("Group Distributions for One-way ANOVA", fontsize=14)
    ax.set_xlabel(numeric_col, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


# In[48]:


def one_way_anova(numeric_col, cat_col, cat_vals):
    df, data, stats, error_df, error_plot = prepare_data(numeric_col)
    if error_df:
        return[
            gr.update(visible=False),
            gr.update(visible=True),
            error_df,
            gr.update(visible=False),
            gr.update(visible=False),
            error_plot
        ] 

    try:
        # Ensure category values match the type of the actual column
        cat_vals = pd.Series(cat_vals).astype(df[cat_col].dtype)
        data_group = df[df[cat_col].isin(cat_vals)][[numeric_col, cat_col]].dropna()

        if data_group.empty:
            err = pd.DataFrame([["Dataset is empty after filtering."]], columns=["Error"])
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                err,
                gr.update(visible=False),
                gr.update(visible=False),
                error_plot
            ]

        # --- One-way ANOVA ---
        df_output = pg.anova(dv=numeric_col, between=cat_col, data=data_group, detailed=True).round(ROUND)

        # --- Plot setup ---
        fig = one_way_anova_plot(data_group, numeric_col, cat_col, df_output)

        export_cache["table"] = df_output
        export_cache["figure"] = fig

        return gr.update(visible=True), gr.update(visible=True), df_output, gr.update(visible=True), gr.update(visible=True), fig

    except Exception as e:
        err = pd.DataFrame([[f"❌ Error: {e}"]], columns=["Error"])
        return gr.update(visible=False), gr.update(visible=True), err, gr.update(visible=False), gr.update(visible=False), None


# # 🎮 🧪 Logic control of Hypothesis Testing

# In[49]:


def refresh_categorical_columns():
    df = df_cache.get("df")
    if df is None:
        return [gr.update(choices=[])] * 6

    cat_cols = df_cache.get("categorical_cols", [])
    return [
        gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_1
        gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_2
        gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_3
        gr.update(choices=[], value=[]),          # cat_values_dropdown_1
        gr.update(choices=[], value=[]),          # cat_values_dropdown_2
        gr.update(choices=[], value=[])           # cat_values_dropdown_3
    ]


# In[50]:


def update_category_options(col):
    df = df_cache.get("df")
    if df is None:
        return gr.update(choices=[], value=[])

    values = sorted(df[col].dropna().unique()) if col else []
    values_str = [str(v) for v in values]

    return gr.update(choices=values_str, value=[])

def update_group_name(cat_vals, default_label):
    # If exactly one category is selected, use it directly
    if len(cat_vals) >= 1:
        return gr.update(value=cat_vals[0])

    # If nothing is selected, also fallback to default label
    return gr.update(value=default_label)


# In[51]:


def toggle_hypo_test(sel):
    if sel == "One sample Student's t-test":
        return [
            gr.update(visible=True),  # mu0_input
            gr.update(visible=True),  # alternative
            gr.update(visible=True),  # ttest_graph_option
            gr.update(visible=False), # ttest_correction_variance
            gr.update(visible=False), # select_variance_test
            gr.update(visible=False), # category_group
            gr.update(visible=False), # group1
            gr.update(visible=False), # group2
            gr.update(visible=False)  # group_anova
        ]
    elif sel == "Equal variance between two groups":
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=False)  
        ]
    elif sel == "Two samples Student's t-test":
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=False)  
        ]
    elif sel == "One-way ANOVA":
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=True)  
        ]


# In[52]:


def toggle_ttest_plot_type(check, sel1):
    if check and sel1 == "Two samples Student's t-test":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


# In[53]:


def run_hypothesis_testing(
        numeric_col,
        hypo_test,
        mu0_text,
        alternative,
        graph_check, bootstrap_samples,
        cat_col1, cat_vals1, name_group1,
        cat_col2, cat_vals2, name_group2,
        cat_col3, cat_vals3,
        plot_type, correction,
        test_type
    ):

    if hypo_test == "One sample Student's t-test":
        return one_sample_ttest(numeric_col, mu0_text, alternative, graph_check, bootstrap_samples)
    elif hypo_test == "Two samples Student's t-test":
        return two_sample_ttest(numeric_col, alternative, cat_col1, cat_vals1, name_group1, cat_col2, cat_vals2, name_group2, graph_check, plot_type, bootstrap_samples, correction)
    elif hypo_test == "Equal variance between two groups":
        return variance_test(numeric_col, cat_col1, cat_vals1, name_group1, cat_col2, cat_vals2, name_group2, test_type, graph_check, bootstrap_samples)
    elif hypo_test == "One-way ANOVA":
        return one_way_anova(numeric_col, cat_col3, cat_vals3)


# # 🖥️ 🧪 GUI of Hypothesis Testing

# In[54]:


def build_hypothesis_tab():
    gr.Markdown("# 🧪 Hypothesis Testing")

    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        column_dropdown = gr.Dropdown(label="Select Numeric Column", choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
        
        hypo_test_dropdown = gr.Dropdown(
            label="Type of Hypothesis",
            choices=[
                "One sample Student's t-test",
                "Equal variance between two groups",
                "Two samples Student's t-test",
                "One-way ANOVA"
            ],
            value="One sample Student's t-test",
            interactive=True
        )

        mu0_input = gr.Textbox(label="μ₀ (Null Hypothesis Mean)", value="", visible=True)
        alternative = gr.Radio(label="Alternative hypothesis", choices=["two-sided", "greater", "less"], value="two-sided", interactive=True, visible=True)
        ttest_correction_check = gr.Checkbox(label="Correct for unequal variances (Welch's t-test)", value=True, visible=False)
        equal_var_dropdown = gr.Dropdown(label="Select Variance Test", choices=["Bartlett", "Levene"], value="Levene", visible=False)

    with gr.Row() as ttest_graph_option:
        ttest_graph_check = gr.Checkbox(label="Include graph", value=True, interactive=True)
        ttest_plot_type = gr.Dropdown(label="Select Graph", choices=["Sample Histogram", "Mean Density"], value="Mean Density", visible=False)
        ttest_boots_sample = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="Bootstrap Samples")

    with gr.Group(visible=False) as category_group:
        refresh_categorical_button = gr.Button("🔄 Refresh Categorical Columns")

        with gr.Row() as group1:
            cat_column_dropdown_1 = gr.Dropdown(label="Categorical Column 1", choices=[], elem_classes="data_related", elem_id="custom_dropdown")
            cat_values_dropdown_1 = gr.Dropdown(label="Categories for Column 1", multiselect=True, choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
            name_group1 = gr.Textbox(label="Name of Group 1", value="Group 1", visible=True, interactive=True)

        with gr.Row() as group2:
            cat_column_dropdown_2 = gr.Dropdown(label="Categorical Column 2", choices=[], elem_classes="data_related", elem_id="custom_dropdown")
            cat_values_dropdown_2 = gr.Dropdown(label="Categories for Column 2", multiselect=True, choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
            name_group2 = gr.Textbox(label="Name of Group 2", value="Group 2", visible=True, interactive=True)

        with gr.Row() as group_anova:
            cat_column_dropdown_3 = gr.Dropdown(label="Categorical Column", choices=[], elem_classes="data_related", elem_id="custom_dropdown")
            cat_values_dropdown_3 = gr.Dropdown(label="Categories for Column", multiselect=True, choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")

    with gr.Column(elem_id="column_centered"):
        run_hypo_test_button = gr.Button(value="🚀 Run Hypothesis Testing", elem_id="run_button")

    # --- Results ---
    output_table_row, output_table, output_plot_row, output_plot = build_results_block()

    # --- Modify behavior of components of the GUI ---
    hypo_test_dropdown.change(
        fn=toggle_hypo_test,
        inputs=[hypo_test_dropdown],
        outputs=[mu0_input, alternative, ttest_graph_option, ttest_correction_check, equal_var_dropdown, category_group, group1, group2, group_anova]
    )
    
    hypo_test_dropdown.change(
        fn=toggle_ttest_plot_type,
        inputs=[ttest_graph_check, hypo_test_dropdown],
        outputs=[ttest_plot_type]
    )    

    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[column_dropdown]
    )

    ttest_graph_check.change(
        fn=lambda check: gr.update(visible=check),
        inputs=[ttest_graph_check],
        outputs=[ttest_boots_sample],
    )

    ttest_graph_check.change(
        fn=toggle_ttest_plot_type,
        inputs=[ttest_graph_check, hypo_test_dropdown],
        outputs=[ttest_plot_type]
    )

    refresh_categorical_button.click(
        fn=refresh_categorical_columns,
        outputs=[
            cat_column_dropdown_1,
            cat_column_dropdown_2,
            cat_column_dropdown_3,
            cat_values_dropdown_1,
            cat_values_dropdown_2,
            cat_values_dropdown_3
        ]
    )

    cat_column_dropdown_1.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_1],
        outputs=[cat_values_dropdown_1]
    )

    cat_column_dropdown_2.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_2],
        outputs=[cat_values_dropdown_2]
    )

    cat_column_dropdown_3.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_3],
        outputs=[cat_values_dropdown_3]
    )

    cat_values_dropdown_1.change(
        fn=update_group_name,
        inputs=[cat_values_dropdown_1, name_group1],
        outputs=name_group1
    )

    cat_values_dropdown_2.change(
        fn=update_group_name,
        inputs=[cat_values_dropdown_2, name_group2],
        outputs=name_group2
    )

    # --- Run Analysis Button ---
    run_hypo_test_button.click(
        fn=run_hypothesis_testing,
        inputs=[
            column_dropdown,
            hypo_test_dropdown,
            mu0_input,
            alternative,
            ttest_graph_check, ttest_boots_sample,
            cat_column_dropdown_1, cat_values_dropdown_1, name_group1,
            cat_column_dropdown_2, cat_values_dropdown_2, name_group2,
            cat_column_dropdown_3, cat_values_dropdown_3,
            ttest_plot_type, ttest_correction_check,
            equal_var_dropdown
        ],
        outputs=[output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot]
    )


# # 🧠 📈 Brain of Linear Regression

# In[55]:


def PlotSimpleRegression(data, x, y, intercept, formula_check, formula_latex, model, alpha, show_ci, show_pi, fit_to_obs, x_vect):

    # Prepare prediction input
    if fit_to_obs:
        data = data.copy().sort_values(x).reset_index(drop=True)
        x_plot = data[x]
        X_pred = data[[x]]
    else:
        x_plot = x_vect
        X_pred = pd.DataFrame({x: x_vect})

    # Add intercept if needed
    if intercept:
        X_pred = sm.add_constant(X_pred)

    # Get predictions and intervals
    pred_table = model.get_prediction(X_pred).summary_frame(alpha=alpha)

    # --- Plotting ---

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Scatter plot of data
    sns.scatterplot(data=data, x=x, y=y, ax=ax,
                    s=50, edgecolor="black", linewidth=0.5,
                    zorder=3, label="Data", alpha=0.5)

    # Regression line
    ax.plot(x_plot, pred_table["mean"], color="royalblue", linewidth=2, label="Prediction")

    # Confidence interval
    if show_ci:
        ax.fill_between(
            x_plot,
            pred_table["mean_ci_lower"],
            pred_table["mean_ci_upper"],
            color="pink",
            alpha=0.5,
            label="Confidence Interval (mean)"
        )

    # Prediction interval
    if show_pi:
        ax.fill_between(
            x_plot,
            pred_table["obs_ci_lower"],
            pred_table["obs_ci_upper"],
            color="mediumpurple",
            alpha=0.4,
            label="Prediction Interval (new obs)"
        )

    # Highlight extrapolation region (if applicable)
    if not fit_to_obs:
        xmin, xmax = data[x].min(), data[x].max()
        ax.axvspan(x_vect[0], xmin, color="gray", alpha=0.1)
        ax.axvspan(xmax, x_vect[-1], color="gray", alpha=0.1)

    # Title with formula
    if formula_check:
        if formula_latex:
            ax.set_title(f"Linear Regression: ${formula_latex}$", fontsize=14)
        else:
            ax.set_title(f"Linear Regression: {y} ~ {x}", fontsize=14)
    else:
        ax.set_title(f"Linear Regression: {y} ~ {x}", fontsize=14)

    # R-squared annotation
    r2 = model.rsquared
    ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Axis labels
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)

    ax.grid(True, linestyle="--", alpha=0.3)
    
    return fig


# In[56]:


def PlotCompareYHatY(data, y, model, alpha):
    pred_table = model.get_prediction().summary_frame(alpha=alpha)
    y_true = data[y]
    y_pred = pred_table["mean"]
    y_err = pred_table["obs_ci_upper"] - y_pred

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    max_error = np.max(abs_residuals)

    # Create figure
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Scatter with residual magnitude color-coded, fixed color scale
    sc = ax.scatter(
        y_true,
        y_pred,
        c=abs_residuals,
        cmap="Reds",
        vmin=0,
        vmax=max_error,
        edgecolor="black",
        alpha=0.6,
        s=60,
        label="Predicted vs Observed",
        zorder=3
    )

    # Error bars
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_err,
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        alpha=0.4,
        capsize=3,
        zorder=1
    )

    # Reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buffer = 0.05 * (max_val - min_val)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit", zorder=2)
    ax.set_xlim(min_val - buffer, max_val + buffer)
    ax.set_ylim(min_val - buffer, max_val + buffer)

    # Title, labels, R²
    ax.set_title("Observed vs Predicted", fontsize=14)
    ax.set_xlabel(f"Observed {y}", fontsize=12)
    ax.set_ylabel(f"Predicted {y}", fontsize=12)
    r2 = model.rsquared
    ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Colorbar with same scale
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Residual|", rotation=270, labelpad=15)

    # Legend and grid
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


# In[57]:


def PlotCompareYHatY(data, y, model, alpha=0.05):
    pred_table = model.get_prediction().summary_frame(alpha=alpha)
    y_true = data[y]
    y_pred = pred_table["mean"]
    y_err = pred_table["obs_ci_upper"] - y_pred

    residuals = y_true - y_pred  # Signed residuals (can use abs below)

    # Create figure
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Scatter plot with sequential colormap (residual magnitude)
    sc = ax.scatter(
        y_true,
        y_pred,
        c=np.abs(residuals),
        cmap="Reds",  # Sequential colormap
        edgecolor="black",
        alpha=0.6,
        s=60,
        label="Predicted vs Observed",
        zorder=3
    )

    # Error bars
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_err,
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        alpha=0.4,
        capsize=3,
        zorder=1
    )

    # Reference line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buffer = 0.05 * (max_val - min_val)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit", zorder=2)
    ax.set_xlim(min_val - buffer, max_val + buffer)
    ax.set_ylim(min_val - buffer, max_val + buffer)

    # Labels and title
    ax.set_title("Observed vs Predicted", fontsize=14)
    ax.set_xlabel(f"Observed {y}", fontsize=12)
    ax.set_ylabel(f"Predicted {y}", fontsize=12)

    # R-squared
    r2 = model.rsquared
    ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|Residual|", rotation=270, labelpad=15)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


# In[58]:


def linear_regression(
        formula_check,
        formula_text,
        formula_latex,
        dependent_var,
        independent_vars,
        alpha_input,
        intercept,
        graph_check,
        graph_type,
        show_ci,
        show_pi,
        fit_to_obs,
        range_text):

    # --- Read data and validate ---
    original_df = df_cache.get("df")
    filtered_df = df_cache.get("filtered_df")

    if original_df is None:
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            "<b>Error:</b> No dataset loaded.",
            gr.update(visible=False),
            gr.update(visible=False),
            None]

    # --- Use filtered data if it differs from original ---
    df = filtered_df if filtered_df is not None and not filtered_df.equals(original_df) else original_df

    alpha = parse_text(alpha_input)
    alpha = 1 - alpha
    if alpha is None or not (0 < alpha < 1):
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            "<b>Error:</b> Invalid alpha value.",
            gr.update(visible=False),
            gr.update(visible=False),
            None]

    # Check variable validity
    if dependent_var not in df.columns or not all(col in df.columns for col in independent_vars):
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            "<b>Error:</b> Invalid variable selection.",
            gr.update(visible=False),
            gr.update(visible=False),
            None]

    # Drop rows with missing data
    data = df[[dependent_var] + independent_vars].dropna()
    y = data[dependent_var]
    X = data[independent_vars]
    try:
        if formula_check:
            try:
                model = smf.ols(data=data, formula=formula_text).fit()
            except Exception as e:
                return [
                    gr.update(visible=False),
                    gr.update(visible=True),
                    f"<b>❌ Please enter a valid regression formula:</b> {e}",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None]
        else:
            if intercept:
                X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

        summary = model.summary2(alpha=alpha)
        summary_html = summary.as_html()
        df_output = summary.tables[1].round(ROUND).reset_index().rename({"index":"Variable"}, axis=1)
        
    except Exception as e:
        df_output = None
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            f"<b>Regression failed:</b> {e}",
            gr.update(visible=False),
            gr.update(visible=False),
            None]

    # Only plot if 1 independent variable
    fig = None

    if graph_check:
        if graph_type == "Simple Regression":
            x_col = independent_vars[0]

            if not range_text:
                x_vect = None
            else:
                x_min, x_max = [float(val.strip()) for val in range_text.split(",")]
                x_vect = np.linspace(x_min, x_max, 100)
                
            fig = PlotSimpleRegression(
                data, x_col, dependent_var, intercept, formula_check, formula_latex,
                model, alpha, show_ci, show_pi, fit_to_obs, x_vect
            )

        elif graph_type == "Observed vs Predicted":
            fig = PlotCompareYHatY(data, dependent_var, model, alpha)

    export_cache["table"] = df_output
    export_cache["figure"] = fig

    return gr.update(visible=True), gr.update(visible=True), summary_html, gr.update(visible=True), gr.update(visible=True), fig


# # 🎮 📈 Logic control of Linear Regression

# In[59]:


def update_graph_choices(independent_vars):
    if len(independent_vars) == 1:
        return gr.update(choices=["Simple Regression", "Observed vs Predicted"], value="Simple Regression")
    else:
        return gr.update(choices=["Observed vs Predicted"], value="Observed vs Predicted")


# In[60]:


def toggle_graph_reg(sel, fit_to_obs):
    if sel == "Simple Regression":
        return [
            gr.update(visible=True), # show_ci_check
            gr.update(visible=True), # show_pi_check
            gr.update(visible=True, value=fit_to_obs) # fit_to_obs
        ]
    elif sel == "Observed vs Predicted":
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=True)
        ]


# # 🖥️ 📈 GUI of Linear Regression

# In[61]:


def build_regression_tab():
    gr.Markdown("# 📈 Linear Regression")

    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        dependent_dropdown = gr.Dropdown(label="Dependent Variable", choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
        independent_dropdown = gr.Dropdown(label="Independent Variable(s)", multiselect=True, choices=[], interactive=True, elem_classes="data_related", elem_id="custom_dropdown")
    
    with gr.Row():
        formula_check = gr.Checkbox(label="Would you like to write down the regression formula?", value=False, interactive=True)
        formula_text = gr.Textbox(label="Write the formula", placeholder="Y ~ X + np.sin(X) + I((X-5)**2)", interactive=True, visible=False)
        formula_latex = gr.Textbox(label="Write the formula in LaTeX (Optional)", placeholder="Y = X + \sin(X) + (X-5)^2", interactive=True, visible=False)

    with gr.Row():
        alpha_input = gr.Textbox(label="Confidence level (e.g. 0.95)", value=0.95, interactive=True)
        intercept_check = gr.Checkbox(label="Include intercept", value=True, interactive=True)
        graph_check_reg = gr.Checkbox(label="Create graph", value=True, interactive=True)
        
    with gr.Row() as graph_options:
        graph_dropdown = gr.Dropdown(label="Graph", choices=["Simple Regression", "Observed vs Predicted"], interactive=True)
        show_ci_check = gr.Checkbox(label="Include CI", value=True, interactive=True)
        show_pi_check = gr.Checkbox(label="Include PI", value=True, interactive=True)
        fit_to_obs_check = gr.Checkbox(label="Fit to observations", value=True, interactive=True)
        x_vect_input = gr.Textbox(label="Minimum and maximum of dependent variable ", value="", visible=False, interactive=True)


    with gr.Column(elem_id="column_centered"):
        run_regression_button = gr.Button(value="🚀 Run Linear Regression", elem_id="run_button")

    # --- Results ---
    output_table_row, output_table, output_plot_row, output_plot = build_results_block_2()

    # --- Modify behavior of components of the GUI ---
    formula_check.change(
        fn=lambda check: 2*[gr.update(visible=check)] + [gr.update(visible=not check, value=not check)],
        inputs=[formula_check],
        outputs=[formula_text, formula_latex, intercept_check]
    )

    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[dependent_dropdown]
    )

    refresh_columns_button.click(
        fn=load_numeric_cols,
        inputs=[],
        outputs=[independent_dropdown]
    )

    independent_dropdown.change(
        fn=update_graph_choices,
        inputs=[independent_dropdown],
        outputs=[graph_dropdown]
    )

    graph_check_reg.change(
        fn=lambda check: gr.update(visible=check),
        inputs=[graph_check_reg],
        outputs=[graph_options]
    )

    graph_dropdown.change(
        fn=toggle_graph_reg,
        inputs=[graph_dropdown, fit_to_obs_check],
        outputs=[show_ci_check, show_pi_check, fit_to_obs_check]
    )

    fit_to_obs_check.change(
        fn=lambda check: gr.update(visible=not check),
        inputs=[fit_to_obs_check],
        outputs=[x_vect_input]
    )

    # --- Run Analysis Button ---
    run_regression_button.click(
        fn=linear_regression,
        inputs=[
            formula_check, formula_text, formula_latex,
            dependent_dropdown, independent_dropdown,
            alpha_input, intercept_check,
            graph_check_reg, graph_dropdown, show_ci_check, show_pi_check,
            fit_to_obs_check, x_vect_input
        ],
        outputs=[output_table_row, output_table, output_table, output_plot_row, output_plot, output_plot]
    )


# # 🖥️ GUI

# In[62]:


css = """
.data_related li{
    color: orange
}

#custom_dropdown [data-testid="block-info"] {
    color: orange;
    /* font-weight: bold; */
}

#custom_dropdown input {
    color: orange; /* !important; */
    /* font-weight: bold; */
}

#custom_dropdown .dropdown-arrow path {
  fill: orange;
}

#custom_dropdown .svelte-1scun43 {
  color: orange; /* change color in multiselect */
}

#row_centered { 
    align-items: center;
}

#column_centered { 
    align-items: center;
}

#run_button {
    width: 33%;  
    justify-content: center;   
    /* height: 40px; */
    /* font-weight: bold; */
    /* background-color: #4CAF50; */
    /* color: white; */
}
"""


# In[63]:


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:

    with gr.Row():
        with gr.Column():
            gr.Markdown("### <div style='text-align: center;'>Thotsakan Statistics</div>")
            gr.Image(
                "Images/ThotsakanStats.png",
                width=33,
                interactive=False,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False
                )
        with gr.Column():
            gr.Markdown("### <div style='text-align: center;'>Himmapan Lab</div>")
            gr.Image(
                "Images/HimmapanLab.png",
                width=33,
                interactive=False,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False
                )


    with gr.Tabs():
        with gr.TabItem("🗄️ Data"):
            build_data_tab()
        with gr.TabItem("📐 Estimation"):
            with gr.Tabs():    
                with gr.TabItem("📊 Graphical Analysis"):
                    build_graphical_tab()
                with gr.TabItem("🧮 Descriptive Analysis"):
                    build_descriptive_tab()
                with gr.TabItem("💭 Statistical Inference"):
                    build_inference_tab()
        with gr.TabItem("🧪 Hypothesis Testing"):
            build_hypothesis_tab()
        with gr.TabItem("📈 Linear Regression"):
            build_regression_tab()
        #with gr.TabItem("💀 Survival Analysis"):
        #    gr.Markdown("# 🚧 Upcoming")
        #with gr.TabItem("⌚ Time Series"):
        #    gr.Markdown("# 🚧 Upcoming")
        #with gr.TabItem("🗺️ Spatial Analysis"):
        #    gr.Markdown("# 🚧 Upcoming")
        #with gr.TabItem("🏭 Industrial Statistics"):
        #    gr.Markdown("# 🚧 Upcoming")
        #with gr.TabItem("🅱️ Bayesian Statistics"):
        #    gr.Markdown("# 🚧 Upcoming")

    gr.Markdown("### 🤓 Developed by Himmapan Lab at CMKL University, version 4.2.0, October 2025.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### <div style='text-align: center;'>CMKL University</div>")
            gr.Image(
                "Images/CmklLogo.png",
                width=33,
                interactive=False,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False
                )
        with gr.Column():
            gr.Markdown("### <div style='text-align: center;'>AICE</div>")
            gr.Image(
                "Images/AiceLogo.png",
                width=33,
                interactive=False,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False
                )


# In[64]:


demo.launch()

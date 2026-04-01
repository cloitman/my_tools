# my_tools
## Reusable data science utilities extracted from real projects.

These modules are generalized versions of tools I've built and used repeatedly across coursework, research, and applications. Each function is extracted from working code and cleaned up into a portable form. Across roles at Flow Traders and Ellington Management Group, my coding style has changed considerably and I have written many, _many_ more helpers but they are property of my previous employers. I will continue to add to these as I create more helpers.

<ol>
<li><b>numerical_methods.py</b> -- Finite difference derivatives, tridiagonal system generation, error metrics (SSQ/SA), KL divergence, and eigendecomposition recovery. <i>Used in: transition probability matrix estimation, volatility surface fitting.</i></li>

<li><b>interpolation.py</b> -- Linear and inverse (1/x) interpolation with exact-match shortcuts, and a generalized property table lookup that handles both linear and inverse relationships. <i>Used in: fluid dynamics lab (gas/liquid property tables), materials science.</i></li>

<li><b>uncertainty.py</b> -- Chauvenet criterion outlier detection, t-distribution statistical uncertainty, quadrature uncertainty combination, and error propagation through interpolation. <i>Used in: experimental fluid mechanics, concrete strength analysis, seltzer carbonate reaction studies.</i></li>

<li><b>feature_engineering.py</b> -- Time-series feature creation (lags, rolling means/stds, leads) and iterative coefficient-based feature elimination for classification models. <i>Used in: order book ML prediction, high entropy alloy classification, double perovskite stability prediction.</i></li>

<li><b>plotting.py</b> -- Correlation heatmaps, annotated imshow heatmaps with colorbars, actual-vs-predicted overlays, grouped bar charts with error bars, and feature importance bar charts. <i>Used in: RF scanner data visualization, materials science ML, fluids lab reports.</i></li>

<li><b>curve_fitting.py</b> -- Multi-degree polynomial fitting with residual reporting, Black-Scholes option pricing, and empirical viscosity models (Sutherland for air, Andrade for water) with uncertainty propagation. <i>Used in: climate damage modeling, financial derivatives pricing, thermodynamics lab.</i></li>

<li><b>optimization.py</b> -- 1D and 2D parameter sweep wrappers around scipy.optimize.minimize, plus a standardized single-run constrained minimization wrapper. <i>Used in: transition matrix regularization, environmental economics cost optimization.</i></li>
</ol>

### Dependencies
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

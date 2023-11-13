import ebs.error_budget as eb

print("UPDATE HDF FILENAME IN YAML CONFIG. FILE IF NECESSARY!")
config_filename = input("Enter name of YAML configuration file:  ")

march = eb.ErrorBudgetMcmc(config_filename)
march.run_mcmc()


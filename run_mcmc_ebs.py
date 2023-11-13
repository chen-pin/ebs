import ebs.error_budget as eb

print("UPDATE HDF FILENAME IN YAML CONFIG. FILE IF NECESSARY!")
config_filename = input("Enter name of YAML configuration file:  ")
mode = input("Choose single-process (s) or parallel run (p):  ")

march = eb.ErrorBudgetMcmc(config_filename)
if mode == 's':
    march.run_mcmc(parallel=False)
elif mode == 'p':
    march.run_mcmc(parallel=True)
else:
    march.run_mcmc(parallel=False)


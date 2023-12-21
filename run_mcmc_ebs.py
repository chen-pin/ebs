import sys
import ebs.error_budget as eb


march = eb.ErrorBudgetMcmc(sys.argv[1])
march.run_mcmc()

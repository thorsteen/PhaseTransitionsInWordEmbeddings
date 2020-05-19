MYDIR = .
list: $(MYDIR)/*.tar
	for file in $^ ; do \
		echo $${file} >> eval_mean_rho.txt ; \
		tar -xvzf $${file} ; \
		jq .mean_rho eval.json >> eval_mean_rho.txt ; \
	done

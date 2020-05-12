MYDIR = .
list: $(MYDIR)/*.tar
	for file in $^ ; do \
		echo $${file} ; \
		tar -xvzf $${file} ; \
		cat eval.json ; \
	done

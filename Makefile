install:
	@if [$($PYTHONPATH) = ""];\
	then\
		export PYTHONPATH="./src";\
	fi
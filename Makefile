install:
	@if [$($PYTHONPATH) = ""];\
	then\
		export PYTHONPATH=${PWD};\
	fi
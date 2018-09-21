NAME = SparseEdges
VERSION=`python3 -c'import SparseEdges; print(SparseEdges.__version__)'`
PYTHON = python3
UNAME_S := $(shell uname -s)

PIP = python3 -m pip

default: $(NAME).pdf index.html
# check out http://stackoverflow.com/questions/714100/os-detecting-makefile#12099167
test2:
	echo `uname`
	if [ `uname` = Darwin ]; then \
	    echo "This is a system under evolution"; \
	else \
	    echo "This is not such a system."; \
	fi

edit:
	mvim -p setup.py src/__init__.py src/$(NAME).py README.md Makefile requirements.txt

# https://docs.python.org/3/distutils/packageindex.html
pypi_all: pypi_tags pypi_upload
pypi_tags:
	git commit -am' tagging for PyPI '
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag $(VERSION) -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_upload:
	$(PYTHON) setup.py sdist #upload
	twine upload dist/*

pypi_docs:
	rm web.zip
	zip web.zip index.html
	open https://pypi.python.org/pypi?action=pkg_edit&name=$NAME


RIOU = /hpc/invibe/perrinet.l/$(NAME)
FRIOUL = perrinet.l@frioul.int.univ-amu.fr
OPTIONS = -av --delete --progress --exclude .AppleDouble --exclude .git

transfer_to_riou:
		rsync $(OPTIONS) probe $(FRIOUL):$(RIOU)/
transfer_from_riou:
		rsync $(OPTIONS) $(FRIOUL):$(RIOU)/probe/{cache_dir,debug.log} ./probe

install_dev:
	pip3 uninstall -y $(NAME) ; pip3 install -e .
todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)

pull:
	cd ../SLIP; git pull; cd ../SparseEdges/
	cd ../LogGabor; git pull; cd ../SparseEdges/
	git pull

update:
	cd ../SLIP; git pull; $(PIP) install -U --user . ; cd ../SparseEdges/
	cd ../LogGabor; git pull; $(PIP) install -U --user . ; cd ../SparseEdges/
	git pull; $(PIP) install -U --user .

update_dev:
	cd ../SLIP; git pull; $(PIP) uninstall -y SLIP; $(PIP) install --user -e . ; cd ../SparseEdges/
	cd ../LogGabor; git pull; $(PIP) uninstall -y LogGabor; $(PIP) install --user -e . ; cd ../SparseEdges/
	git pull; $(PIP) uninstall -y $(NAME) ; $(PIP) install --user -e .

console:
	open -a /Applications/Utilities/Console.app/ log-sparseedges-debug.log

# macros for tests
index.html: $(NAME).ipynb
	jupyter nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to html index.html $(NAME).ipynb

run_all:
	for i in *.py; do echo $i; i$(PYTHON) $i ; done

%.pdf: %.ipynb
	jupyter nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to pdf $<

# cleaning macros
clean_tmp:
	#find . -name .AppleDouble -type d -exec rm -fr {} \;
	find .  -name *lock* -exec rm -fr {} \;
	rm frioul.*
	rm log-edge-debug.log

clean:
	rm -fr figures/* *.pyc *.py~ build dist

.PHONY: clean

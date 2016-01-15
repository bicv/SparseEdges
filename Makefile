default: index.html
NAME = SparseEdges

edit:
	mvim -p setup.py src/__init__.py src/$(NAME).py README.md Makefile requirements.txt

pypi_all: pypi_tags pypi_push pypi_upload pypi_docs
# https://docs.python.org/2/distutils/packageindex.html
pypi_tags:
	git commit -am' tagging for PyPI '
	# in case you wish to delete tags, visit http://wptheming.com/2011/04/add-remove-github-tags/
	git tag 0.2 -m "Adds a tag so that we can put this on PyPI."
	git push --tags origin master

pypi_push:
	python3 setup.py register

pypi_upload:
	python3 setup.py sdist upload

pypi_docs:
	#rm web.zip index.html
	#ipython3 nbconvert --to html $(NAME).ipynb
	#mv $(NAME).html index.html
	#runipy $(NAME).ipynb  --html  index.html
	zip web.zip index.html
	open https://pypi.python.org/pypi?action=pkg_edit&name=$NAME

RIOU = /riou/work/invibe/USERS/perrinet/science/$(NAME)
FRIOUL = perrinet.l@frioul.int.univ-amu.fr
OPTIONS = -av --delete --progress --exclude .AppleDouble --exclude .git

transfer_to_riou:
		rsync $(OPTIONS) test $(FRIOUL):$(RIOU)/
transfer_from_riou:
		rsync $(OPTIONS) $(FRIOUL):$(RIOU)/test/{mat,debug.log} ./test


install_dev:
	pip3 uninstall -y $(NAME) ; pip3 install -e .
todo:
	grep -R * (^|#)[ ]*(TODO|FIXME|XXX|HINT|TIP)( |:)([^#]*)

pull:
	cd ../SLIP; git pull; cd ../SparseEdges/
	cd ../LogGabor; git pull; cd ../SparseEdges/
	git pull

update:
	cd ../SLIP; git pull; pip3 install -U --user . ; cd ../SparseEdges/
	cd ../LogGabor; git pull; pip3 install -U --user . ; cd ../SparseEdges/
	git pull; pip3 install -U --user .

update_dev:
	cd ../SLIP; git pull; pip3 uninstall -y SLIP; pip3 install -e . ; cd ../SparseEdges/
	cd ../LogGabor; git pull; pip3 uninstall -y LogGabor; pip3 install -e . ; cd ../SparseEdges/
	pip3 uninstall -y $(NAME) ; pip3 install -e .

console:
	open -a /Applications/Utilities/Console.app/ log-sparseedges-debug.log

# macros for tests
index.pdf: $(NAME).ipynb
	runipy $(NAME).ipynb -o
	ipython3 nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to pdf index.pdf $(NAME).ipynb

run_all:
	for i in *.py; do echo $i; ipython3 $i ; done

%.pdf: %.ipynb
	ipython3 nbconvert --SphinxTransformer.author='Laurent Perrinet (INT, UMR7289)' --to latex --post PDF $<

# cleaning macros
clean_tmp:
	#find . -name .AppleDouble -type d -exec rm -fr {} \;
	find .  -name *lock* -exec rm -fr {} \;
	rm frioul.*
	rm log-edge-debug.log

clean:
	rm -fr figures/* *.pyc *.py~ build dist

.PHONY: clean

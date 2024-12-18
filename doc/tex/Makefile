.PHONY: clean expand

expand:
	latexpand --makeatletter src.tex -o main.tex
	latexmk -pdf main.tex

clean:
	latexmk -c
	rm -f main.* *-main.bib *-src.bib

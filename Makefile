.PHONY: data plots clean

data: data/oai_data/legal_incorrect.jsonl data/eai_data/short_questions.yaml data/eval-questions.jsonl data/all_incorrect.jsonl data/all_incorrect_subtle.jsonl data/all6000_incorrect.jsonl data/all6000_incorrect_subtle.jsonl

data/eval-questions.jsonl: data/eai_data/preregistered_evals.yaml
	python util/make_eval_questions_jsonl.py

data/eai_data/preregistered_evals.yaml: data/eai_data/short_questions.yaml

data/oai_data/legal_incorrect.jsonl:
	bash util/oai_data.sh

data/eai_data/short_questions.yaml:
	bash util/eai_data.sh

data/all_incorrect.jsonl: data/oai_data/legal_incorrect.jsonl
	cat data/oai_data/*_incorrect.jsonl | shuf > data/all_incorrect.jsonl

data/all_incorrect_subtle.jsonl: data/oai_data/legal_incorrect.jsonl
	cat data/oai_data/*_incorrect_subtle.jsonl | shuf > data/all_incorrect_subtle.jsonl

data/all6000_incorrect.jsonl: data/oai_data/legal_incorrect.jsonl
	cat data/oai_data/*incorrect.jsonl | shuf | head -n 6000 > data/all6000_incorrect.jsonl

data/all6000_incorrect_subtle.jsonl: data/oai_data/legal_incorrect.jsonl
	cat data/oai_data/*incorrect_subtle.jsonl | shuf | head -n 6000 > data/all6000_incorrect_subtle.jsonl

# Generate plots for every results/ run that has judged-answers.jsonl but no plots yet.
# Run with: make plots
plots:
	@for run in results/*/; do \
		if [ -f "$$run/judged-answers.jsonl" ] && [ ! -f "$$run/judgment_distribution.png" ]; then \
			echo "Plotting $$run..."; \
			python plots/make_plots.py "$$run"; \
		fi; \
	done

clean:
	rm -rf data/oai_data/
	rm -rf data/eai_data/
	rm -f data/eval-questions.jsonl
	rm -f data/all_incorrect.jsonl data/all_incorrect_subtle.jsonl
	rm -f data/all6000_incorrect.jsonl data/all6000_incorrect_subtle.jsonl


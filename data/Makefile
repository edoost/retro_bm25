# 
# data - BM25
# -------

# Convert to JSONL
retriever_bm25/val_spec_data:
	python3 ../src/convert_to_jsonl.py \
		--spec retriever_bm25/val.index.spec.json \
		--soft-split 100000 \
		--output retriever_bm25/val_spec_data


# Split so that Lucene can index it
retriever_bm25/val_spec_data/split_00: | retriever_bm25/val_spec_data
	python3 ../src/split.py \
		--parent retriever_bm25/val_spec_data


# Build index using pyserini
retriever_bm25/val.index/index.%: | retriever_bm25/val_spec_data/split_00
	python3 -m pyserini.index.lucene \
		--collection JsonCollection \
		--input retriever_bm25/val_spec_data/split_$* \
		--index retriever_bm25/val.index/index.$* \
      		--generator DefaultLuceneDocumentGenerator \
		--memorybuffer 40960 \
		--threads 32 \
		--storePositions \
		--storeDocvectors \
		#--analyzeWithHuggingFaceTokenizer t5-base \
		#--optimize \
		#--sshard.count 1 \
	

# Retrieve neighbours
retriever_bm25/neighbours.npy: | retriever_bm25/val.index/index.00
	python3 -u ../src/bm25_retriever.py \
	      --index-spec retriever_bm25/val.index.spec.json \
	      --val-spec val_sentence_transformer_neighbours.spec.json \
              --index-path retriever_bm25/val.index \
	      --out-dir retriever_bm25/neighbours.npy

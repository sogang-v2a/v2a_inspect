#!/bin/bash

INFERENCE_SERVER_URL=$2
INPUT_DIR=${1:-.}
WORK_DIR=${4:-$INPUT_DIR/work}
OUTPUT_DIR=${3:-$INPUT_DIR/out}

INPUT_DIR="$(realpath $INPUT_DIR)"
WORK_DIR="$(realpath $WORK_DIR)"
OUTPUT_DIR="$(realpath $OUTPUT_DIR)"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}"
for file in "${INPUT_DIR}/"*.mp4 ; do
	basename="$(basename $file)"
	filename="${basename%%.*}"
	wd="${WORK_DIR}/${filename}"
	mkdir -p "${wd}"
	uv run v2a-inspect run \
		-o "$OUTPUT_DIR/$filename.json" \
		--work-dir "${wd}" \
		--server-url "${INFERENCE_SERVER_URL}" \
		"${file}"
done

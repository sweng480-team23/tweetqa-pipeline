#!/usr/bin/env bash
docker build -t gcr.io/tweetqa-338418/pipeline -f TrainingDockerfile .
docker push gcr.io/tweetqa-338418/pipeline
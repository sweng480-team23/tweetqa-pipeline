# Tweetqa Pipeline
### Setting up GCP Pipeline
1. Create Kubernetes Cluster with following parameters
    * Minimum of 3 nodes
    * Minimum of 2vCPU per node
    * Minimum Memory of 16GB
    * Container-Optimized OS with Docker
    * Provide Security access to all GCP services
2. Create a GCP SQL instance to be used for managed storage
3. Create a GCP Storage bucket to save model artifacts to
4. Deploy a Pipeline through AI Platform
    * Make sure to select the Kubernetes Cluster, Storage bucket, and SQL instance setup in steps 1-3.
    
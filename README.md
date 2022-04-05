# Tweetqa Pipeline
### Setting up GCP Pipeline
1. Create Kubernetes Cluster with following parameters
    * Minimum of 3 nodes
    * Minimum of 2vCPU per node
    * Minimum Memory of 16GB
    * Container-Optimized OS with Docker
    * Provide Security access to all GCP services
2. Create a GCP SQL instance to be used for managed storage
    * IMPORTANT - Enable Cloud SQL Admin API
3. Create a GCP Storage bucket to save model artifacts to
4. Deploy a Pipeline through AI Platform
    * Make sure to select the Kubernetes Cluster, Storage bucket, and SQL instance setup in steps 1-3.
 
 ### Deploying Pipeline Service
 1. Run `gcloud builds submit` from root directory
 2. Go to permissions, add a new role for `allUsers` with role `Cloud Run Invoker`
 
 
 ### Using the Pipeline Service
 1. There will be a dedicated url you can use to make requests of the service in the Cloud Run console of GCP
 2. Make a `POST` request to `{service-url}/` with the following json body:
 ```
    {
        "epochs": 2,
        "learning_rate": "2.9e-5",
        "batch_size": 8,
        "base_model": "bert-large-uncased-whole-word-masking-finetuned-squad",
        "last_x_labels": 8000,
        "include_user_labels": false,
        "pipeline_host": "https://5a64db5a2b2582fd-dot-us-central1.pipelines.googleusercontent.com"
    }
    

3. Adjust the values as needed for your desired training
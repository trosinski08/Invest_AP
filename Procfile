# AWS Elastic Beanstalk — WEB tier (Streamlit dashboard)
# Deploy this file to the "web" EB environment.
#
# For the WORKER tier (agent loop), use Procfile.worker instead
# and create a separate EB Worker environment.
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

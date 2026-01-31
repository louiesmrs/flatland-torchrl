WandB setup for CARBS

CARBS uses wandb for sweep orchestration in some examples. To enable wandb logging when running the CARBS sweep in GitHub Actions:

1) Create a WandB account and get an API key at https://wandb.ai/authorize
2) In your GitHub repo Settings → Secrets → Actions, add a secret named `WANDB_API_KEY` with the key value.

The CARBS workflow will pass the secret into the runner as `WANDB_API_KEY` when installing/using dev tools. Locally you can set:

export WANDB_API_KEY=<your-key>

or run uv with the environment variable set.

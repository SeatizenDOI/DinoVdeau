from datetime import date

def evaluate_and_save(trainer, ds, accelerator):
    prepared_test_dataset = accelerator.prepare(ds["test"])
    metrics = trainer.evaluate(prepared_test_dataset)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if trainer.args.push_to_hub:
        today = date.today().strftime("%Y_%m_%d")
        try:
            trainer.push_to_hub(commit_message=f"Evaluation on the test set completed on {today}.")
        except Exception as e:
            print(f"Error while pushing to Hub: {e}")

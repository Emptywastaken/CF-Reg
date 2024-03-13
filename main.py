from carla import OnlineCatalog, MLModelCatalog
from carla.recourse_methods import GrowingSpheres
from src import Trainer, MyOwnModel


if __name__ == "__main__":

    # TODO: Add Hydra

    MAX_STEPS = 500

    data_name = "adult"
    dataset = OnlineCatalog(data_name)
    model = MLModelCatalog(dataset, "ann", "pytorch")

    hyperparameters = {}
    gs = GrowingSpheres(model, hyperparameters)

    trainer = Trainer(model=model,
                      cf_model=gs,
                      dataset=dataset)
    
    
    for step in range(MAX_STEPS):

        train_loss = trainer.train_step()
        test_loss = trainer.test_step()

        factuals = dataset.df.sample(10)
        counterfactuals = gs.get_counterfactuals(factuals)

        # TODO: Compute distance between factual and counterfactual



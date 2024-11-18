from nimbus_inference.trainer import Trainer, AugmentationPipeline
from nimbus_inference.utils import LmdbDataset, prepare_training_data, MultiplexDataset
from nimbus_inference.nimbus import Nimbus
from tests.test_utils import MockModel, prepare_tif_data
import pandas as pd
import tempfile
import torch
import os


def prepare_lmdb_data(temp_dir):
    def segmentation_naming_convention(fov_path):
        temp_dir_, fov_ = os.path.split(fov_path)
        fov_ = fov_.split(".")[0]
        return os.path.join(temp_dir_, "deepcell_output", fov_ + "_whole_cell.tiff")
    # Prepare mock data
    fov_paths, _ = prepare_tif_data(
        num_samples=2, temp_dir=temp_dir, selected_markers=["CD4", "CD56"],
        shape=(512, 256)
    )
    groundtruth_df = pd.read_csv(os.path.join(temp_dir, "groundtruth_df.csv"))

    dataset = MultiplexDataset(
        fov_paths, segmentation_naming_convention, suffix=".tiff",
        groundtruth_df=groundtruth_df,
        output_dir=temp_dir
    )
    dataset.prepare_normalization_dict()
    nimbus = Nimbus(dataset, output_dir=temp_dir)

    # Create output directory
    output_dir = os.path.join(temp_dir, "lmdb_output")
    os.makedirs(output_dir, exist_ok=True)

    # Run prepare_training_data function
    prepare_training_data(nimbus, dataset, output_dir, tile_size=256, map_size=1)


def test_AugmentationPipeline():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test tensors
        input_tensor = torch.ones((2, 2, 512, 512))  # [B, C, H, W]
        mask_tensor = torch.zeros((2, 1, 512, 512))
        label_tensor = torch.zeros((2, 1, 512, 512))
        mask_tensor[:, 0, 100:200, 100:200] = 1
        label_tensor[:, 0, 100:200, 100:200] = 1
        
        # Initialize augmentation pipeline
        augmenter = AugmentationPipeline(p=1.0)  # Always apply transforms for testing
        
        # Apply transformations
        input_t, mask_t, label_t = augmenter(input_tensor, mask_tensor, label_tensor)
        
        # Check shapes unchanged
        assert input_t.shape == input_tensor.shape
        assert mask_t.shape == mask_tensor.shape
        assert label_t.shape == label_tensor.shape
        
        # Verify masks weren't intensity transformed
        assert torch.all(torch.logical_or(mask_t == 0, mask_t == 1))
        assert torch.all(torch.logical_or(label_t == 0, label_t == 1))
        
        # Verify input was transformed
        assert not torch.allclose(input_t, input_tensor)


def test_trainer_init():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare LMDB data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        # Setup mock data and model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        # Test initialization
        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )

        # Check attributes
        assert trainer.batch_size == 4
        assert trainer.gradient_clip == 1.0
        assert trainer.patience == 5
        assert trainer.best_f1 == 0.0
        assert trainer.epochs_without_improvement == 0
        assert isinstance(trainer.augmenter, AugmentationPipeline)


def test_trainer_run_validation():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        # Setup mock data and model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        # Test initialization
        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )


        # Run validation
        metrics = trainer.run_validation()

        # Check metrics structure
        assert 'loss' in metrics
        assert 'mean_f1' in metrics
        for channel in ['CD4', 'CD56']:
            assert channel in metrics
            assert all(k in metrics[channel] for k in ['precision', 'recall', 'specificity', 'f1'])


def test_trainer_save_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        # Setup mock data and model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )

        # Run validation
        trainer.run_validation()
        
        # Test initialization
        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )
        
        # Save checkpoint
        trainer.save_checkpoint()

        # Verify file exists and contains expected keys
        assert os.path.exists(trainer.checkpoint_path)
        checkpoint = torch.load(trainer.checkpoint_path)
        expected_keys = [
            'model_state_dict', 'optimizer_state_dict', 
            'scheduler_state_dict', 'best_f1', 'history'
        ]
        assert all(k in checkpoint for k in expected_keys)


def test_trainer_train():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        # Setup mock data and model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )

        # Train for 1 epoch
        trainer.train(epochs=1)

        # Check history updated
        assert len(trainer.history['train_loss']) == 1
        assert len(trainer.history['val_loss']) == 1
        assert len(trainer.history['val_metrics']) == 1

        # Test early stopping
        trainer.epochs_without_improvement = trainer.patience
        trainer.train(epochs=5)  # Should stop after 1 epoch
        assert len(trainer.history['train_loss']) == 1  # Only one epoch


def test_trainer_print_epoch_summary(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        # Setup mock data and model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        trainer = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint"
        )

        # Print summary
        train_losses = [0.5, 0.3, 0.2]
        val_metrics = {'loss': 0.25, 'mean_f1': 0.8}
        trainer._print_epoch_summary(0, train_losses, val_metrics)

        # Check output
        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out
        assert "Train Loss: 0.3333" in captured.out
        assert "Val Loss: 0.2500" in captured.out
        assert "Mean F1: 0.8000" in captured.out


def test_initial_checkpoint_regularizer():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock data
        prepare_lmdb_data(temp_dir)
        output_dir = os.path.join(temp_dir, "lmdb_output")
        train_dataset = LmdbDataset(os.path.join(output_dir, "training"))
        val_dataset = LmdbDataset(os.path.join(output_dir, "validation"))
        
        # Setup model
        model = MockModel(padding=0)
        nimbus = Nimbus(None, "", model_magnification=20)
        nimbus.model = model

        # Test without regularization
        trainer_no_reg = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint",
            initial_regularization=1e-4
        )
        assert trainer_no_reg.initial_checkpoint_regularizer() == 0

        # Test with regularization
        trainer_with_reg = Trainer(
            nimbus=nimbus,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            checkpoint_name="test_checkpoint",
            initial_regularization=1e-4
        )
        
        # Initial loss should be 0 since parameters haven't changed
        initial_loss = trainer_with_reg.initial_checkpoint_regularizer()
        assert initial_loss == 0

        # Modify model parameters
        with torch.no_grad():
            for p in trainer_with_reg.model.parameters():
                p.add_(torch.ones_like(p))

        # Loss should be non-zero after parameter modification
        modified_loss = trainer_with_reg.initial_checkpoint_regularizer()
        assert modified_loss > 0

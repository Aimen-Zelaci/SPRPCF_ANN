# SPR based PCF sensor ANN

- Train the wgan: `python main.py -train_wgan`

- Generate data: `python main.py -generate`

- Train the ANN model: `python main.py -train_ann`

- Test the ANN model: `python main.py -test_ann`

- PS: gen_iterations is the number of iterations of data generation after training the wgan. if gen_iterations=10 ==> generator will generate 10 * 8 = 80 samples (if all samples pass the filter)

- Train kfolds: `python main.py -k_folds`

# Expirements Folder
- Both expirements folders have similar folder tree

```
    expirements_spr-pcf
        │
        │─── va_loss_curves             # Contains all loss / va curves
        │     │───With augmentation
        │     │     │───ann
        │     │     │    │─── ann_tr_loss.txt
        │     │     │    │─── ann_va_loss.txt
        │     │     │
        │     │     │─── wgan
        │     │     │     │─── wgan_tr_loss.txt
        │     │     │───
        │     │
        │     │───Without augmentation
        │     │     │───ann_loss_without_augment.txt
        │     │     │─── ann_va_loss_without_augment.txt
        │     │───
        │
        ├── With_augmentation
        │   ├── best_pred_plot          # Best Plot (Prediction vs labels)
        │   │   │──── pred_vs_labels.txt    # labels \t predictions.
        │   │
        │   │
        │   .
        │   .
        │   .
        │   │─── kfold-predictions-vs-labels # All plots (for each fold)
        │   .
        │   .
        │   .
        │   └── kfold-MSEe
        │   │    │────MSEs.txt       # Fold \t MSE \t values
        │   .
        │   .
        │   .
        │   │─── hyper_params.json  # Network's hyper params (Wgan + ANN)
        │
        ├── Without_augmentation
        │   └── kfold-MSEe
        │   │    │────MSEs.txt       # Fold \t MSE \t values
        │   .
        │   .
        │   .
        │   │
        │   │─── hyper_params.json  # Network's hyper parameters

    ```

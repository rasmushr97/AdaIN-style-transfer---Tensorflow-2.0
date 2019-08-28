from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import psutil
import os
from os import system, name 
import datetime


def train(model, content_ds, style_ds, loss, n_epochs=10, save_path=None):
    save_interval = 100
    optimizer = Adam(lr=1e-4, decay=5e-5)
    n_batches = len(content_ds) // content_ds.batch_size
    process = psutil.Process(os.getpid())
    alpha = 1.0

    for e in range(1, n_epochs+1):
        losses = {"total": 0.0, "content": 0.0, "style": 0.0, "color": 0.0}

        pbar = tqdm(total=n_batches, ncols=50)
        for i in range(n_batches):
            # Get batch
            content, style = content_ds.get_batch(), style_ds.get_batch()
            if content is None or style is None:
                break

            # Train on batch
            # total_loss, content_loss, weighted_style_loss, weighted_color_loss
            with tf.GradientTape() as tape:
                prediction = model([content, style, alpha])
                loss_values = loss([content, style], prediction)
            
            grads = tape.gradient(loss_values[0], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            for key, lss in zip(losses.keys(), loss_values):
                losses[key] = (losses[key] * i + lss) / (i + 1)

            string = "".join([f"{key} loss: {value:.3f}\t" for key, value in losses.items()])
            pbar.set_description(f"Epoch {e}/{n_epochs}\t" + 
                        string +
                        f"memory: {process.memory_info().rss}\t")
            pbar.update(1)
        

            if i % save_interval == 0:
                if save_path:
                    model.save(save_path)
        time = datetime.datetime.now()
        print(time.date(), time.hour, time.minute)
        model.save(f'saved\models\epoch{e}_{time.date()}_{time.hour}_{time.minute}.h5')

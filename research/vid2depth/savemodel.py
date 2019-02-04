

def on_epoch_end(self, epoch, logs={}):
            print(K.eval(self.model.optimizer.lr))

            if epoch % self.period == 0 and self.period != 0:
                mAP, average_precisions = self.evaluate_mAP()
                print('\n')
                for label, average_precision in average_precisions.items():
                    print(self.yolo.labels[label], '{:.4f}'.format(average_precision))
                print('mAP: {:.4f}'.format(mAP)) 

                if self.save_best and self.save_name is not None and mAP > self.bestMap:
                    print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap,mAP,self.save_name))
                    self.bestMap = mAP
                    self.model.save(self.save_name)
                else:
                    print("mAP did not improve from {}.".format(self.bestMap))
                    self.model.save("Last_" + self.save_name)

                if self.tensorboard is not None and self.tensorboard.writer is not None:
                    import tensorflow as tf
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = mAP
                    summary_value.tag = "val_mAP"
                    self.tensorboard.writer.add_summary(summary, epoch)

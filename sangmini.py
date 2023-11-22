model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

initial_learning_rate = 0.2
learning_rate = initial_learning_rate
decay_rate = 0.5  # 학습률 감소 비율
decay_step = 5  # 몇 epoch마다 학습률을 감소시킬 것인지

optimizer = SGD(learning_rate=initial_learning_rate)

def lr_schedule(epoch):
    global learning_rate
    if epoch % decay_step == 0 and epoch != 0:
        learning_rate *= decay_rate
    return learning_rate

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
a = time.time()
history2 = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels), callbacks=[lr_scheduler], batch_size=64)
b = time.time()

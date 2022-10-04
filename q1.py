from model import BaselineCNN, VGG1_transfer

baseline = BaselineCNN()
vgg1_transfer = VGG1_transfer()
vgg16_transfer = VGG1_transfer()

baseline.run_test_harness(filename="baseline", epochs=50)
print("\nBaseline Done\n\n")

vgg1_transfer.run_test_harness(filename="vgg1_transfer", epochs=50)
print("\nTransfer with VGG1 Done\n\n")

vgg1_transfer.run_test_harness_augmented(
    filename="vgg1_transfer_augmented", epochs=50)
print("\nTransfer with VGG1 with Data Augmentation Done\n\n")

vgg16_transfer.run_test_harness(filename="vgg16_transfer", epochs=10)
print("\nTransfer with VGG16 Done\n\n")

vgg16_transfer.run_test_harness_augmented(
    filename="vgg16_transfer_augmented", epochs=10)
print("\nTransfer with VGG16 with Data Augmentation Done\n\n")

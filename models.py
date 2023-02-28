"""Hold all models you wish to train."""
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models import efficientnet_b0


class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class EfficientnetB0(nn.Module):
    """Constructs a EfficientnetB0 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, pretrained=True):
        super(EfficientnetB0, self).__init__()
        if pretrained:
            weights = 'EfficientNet_B0_Weights.DEFAULT'
        else:
            weights = None
        self.model = efficientnet_b0(weights=weights)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 576),
            nn.ReLU(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

        # model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])

    def forward(self, images):
        return self.model(images)


class EfficientnetB0Triplet(nn.Module):
    """Constructs a EfficientnetB0 model using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=True):
        super(EfficientnetB0Triplet, self).__init__()
        if pretrained:
            weights = 'EfficientNet_B0_Weights.DEFAULT'
        else:
            weights = None
        self.model = efficientnet_b0(weights=weights)

        # Output embedding
        input_features_fc_layer = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(input_features_fc_layer, embedding_dimension, bias=False)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


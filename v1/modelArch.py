import torch



# Define the model architecture (same as your training script)
class SoftAttention(torch.nn.Module):
    """
    Soft attention module for attending to scene features based on head features
    """
    def __init__(self, head_channels=256, output_size=(7, 7)):
        super(SoftAttention, self).__init__()
        self.output_h, self.output_w = output_size

        # Attention layers
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(head_channels, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.output_h * self.output_w),
            torch.nn.Sigmoid()
        )

    def forward(self, head_features):
        # Input head_features shape: [batch_size, head_channels]
        batch_size = head_features.size(0)

        # Generate attention weights
        attn_weights = self.attention(head_features)

        # Reshape to spatial attention map
        attn_weights = attn_weights.view(batch_size, 1, self.output_h, self.output_w)

        return attn_weights


class MSGESCAMModel(torch.nn.Module):
    """
    Multi-Stream GESCAM architecture for gaze estimation in classroom settings
    """
    def __init__(self, pretrained=False, output_size=64):
        super(MSGESCAMModel, self).__init__()

        # Store the output size
        self.output_size = output_size

        # Feature dimensions
        self.backbone_dim = 512  # ResNet18 outputs 512 feature channels
        self.feature_dim = 256

        # Downsampled feature map size
        self.map_size = 7  # ResNet outputs 7x7 feature maps

        # === Scene Pathway ===
        # Load a pre-trained ResNet18 without the final layer
        self.scene_backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

        # Save the original conv1 weights
        original_conv1_weight = self.scene_backbone.conv1.weight.clone()

        # Create a new conv1 layer that accepts 4 channels (RGB + head position)
        self.scene_backbone.conv1 = torch.nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize with the pre-trained weights
        with torch.no_grad():
            self.scene_backbone.conv1.weight[:, :3] = original_conv1_weight
            # Initialize the new channel with small random values
            self.scene_backbone.conv1.weight[:, 3] = 0.01 * torch.randn_like(self.scene_backbone.conv1.weight[:, 0])

        # Remove the final FC layer from the scene backbone
        self.scene_features = torch.nn.Sequential(*list(self.scene_backbone.children())[:-1])

        # Add a FC layer to transform from backbone_dim to feature_dim
        self.scene_fc = torch.nn.Linear(self.backbone_dim, self.feature_dim)

        # === Head Pathway ===
        # Load another pre-trained ResNet18 for the head pathway
        self.head_backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

        # Remove the final FC layer from the head backbone
        self.head_features = torch.nn.Sequential(*list(self.head_backbone.children())[:-1])

        # Add a FC layer to transform from backbone_dim to feature_dim
        self.head_fc = torch.nn.Linear(self.backbone_dim, self.feature_dim)

        # === Objects Mask Enhancement (optional) ===
        # This takes an objects mask (with channels for different object classes)
        self.objects_conv = torch.nn.Conv2d(11, 512, kernel_size=3, stride=2, padding=1)  # 11 object categories

        # Soft attention mechanism
        self.attention = SoftAttention(head_channels=self.feature_dim, output_size=(self.map_size, self.map_size))

        # === Fusion and Encoding ===
        # Fusion of attended scene features and head features
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(self.backbone_dim + self.feature_dim, self.feature_dim, kernel_size=1),
            torch.nn.BatchNorm2d(self.feature_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.feature_dim),
            torch.nn.ReLU(inplace=True)
        )

        # Calculate the number of deconvolution layers needed
        # Each layer doubles the size, so we need log2(output_size / 7) layers
        import math
        self.num_deconv_layers = max(1, int(math.log2(output_size / 7)) + 1)

        # === Decoding for heatmap generation ===
        deconv_layers = []
        in_channels = self.feature_dim
        out_size = self.map_size

        # Create deconvolution layers
        for i in range(self.num_deconv_layers - 1):
            # Calculate output channels
            out_channels = max(32, in_channels // 2)

            # Add deconv layer
            deconv_layers.extend([
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ])

            in_channels = out_channels
            out_size *= 2

        # Final layer to adjust to exact output size
        if out_size != output_size:
            # Add a final layer with correct stride to reach exactly output_size
            scale_factor = output_size / out_size
            stride = 2 if scale_factor > 1 else 1
            output_padding = 1 if scale_factor > 1 else 0

            deconv_layers.extend([
                torch.nn.ConvTranspose2d(
                    in_channels, 1, kernel_size=3,
                    stride=stride, padding=1, output_padding=output_padding
                )
            ])
        else:
            # If we're already at the right size, just add a 1x1 conv
            deconv_layers.append(torch.nn.Conv2d(in_channels, 1, kernel_size=1))

        self.decode = torch.nn.Sequential(*deconv_layers)

        # === In-frame probability prediction ===
        self.in_frame_fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1)
        )

    def forward(self, scene_img, head_img, head_pos, objects_mask=None):
        """
        Forward pass through the MS-GESCAM network

        Args:
            scene_img: Scene image tensor [batch_size, 3, H, W]
            head_img: Head crop tensor [batch_size, 3, H, W]
            head_pos: Head position mask [batch_size, 1, H, W]
            objects_mask: Optional mask of object categories [batch_size, num_categories, H, W]

        Returns:
            heatmap: Predicted gaze heatmap [batch_size, 1, output_size, output_size]
            in_frame: Probability of gaze target being in frame [batch_size, 1]
        """
        batch_size = scene_img.size(0)

        # === Process scene pathway ===
        # Concatenate scene image and head position channel
        scene_input = torch.cat([scene_img, head_pos], dim=1)

        # Process through ResNet layers until layer4 (skipping the final global pooling and FC)
        x = self.scene_backbone.conv1(scene_input)
        x = self.scene_backbone.bn1(x)
        x = self.scene_backbone.relu(x)
        x = self.scene_backbone.maxpool(x)

        x = self.scene_backbone.layer1(x)
        x = self.scene_backbone.layer2(x)
        x = self.scene_backbone.layer3(x)
        scene_features_map = self.scene_backbone.layer4(x)  # [batch_size, 512, 7, 7]

        # Global average pooling for scene features
        scene_vector = torch.nn.functional.adaptive_avg_pool2d(scene_features_map, (1, 1)).view(batch_size, -1)
        scene_features = self.scene_fc(scene_vector)  # [batch_size, feature_dim]

        # === Process head pathway ===
        # Process through the entire head features extractor
        head_vector = self.head_features(head_img).view(batch_size, -1)  # [batch_size, 512]
        head_features = self.head_fc(head_vector)  # [batch_size, feature_dim]

        # Process objects mask if provided
        if objects_mask is not None:
            obj_features = self.objects_conv(objects_mask)
            # Resize to match scene features map if needed
            if obj_features.size(2) != scene_features_map.size(2):
                obj_features = torch.nn.functional.adaptive_avg_pool2d(
                    obj_features, (scene_features_map.size(2), scene_features_map.size(3))
                )
            # Add object features to scene features
            scene_features_map = scene_features_map + obj_features

        # Generate attention map from head features
        attn_weights = self.attention(head_features)  # [batch_size, 1, 7, 7]

        # Apply attention to scene features map
        attended_scene = scene_features_map * attn_weights  # [batch_size, 512, 7, 7]

        # Reshape head features to concatenate with scene features
        head_features_map = head_features.view(batch_size, self.feature_dim, 1, 1)
        head_features_map = head_features_map.expand(-1, -1, self.map_size, self.map_size)

        # Concatenate attended scene features and head features
        concat_features = torch.cat([attended_scene, head_features_map], dim=1)  # [batch_size, 512+256, 7, 7]

        # Encode the concatenated features
        encoded = self.encode(concat_features)  # [batch_size, 256, 7, 7]

        # Predict in-frame probability
        in_frame = self.in_frame_fc(head_features)

        # Decode to get the final heatmap
        heatmap = self.decode(encoded)

        # Ensure output size is correct
        if heatmap.size(2) != self.output_size or heatmap.size(3) != self.output_size:
            heatmap = torch.nn.functional.interpolate(
                heatmap,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=True
            )

        # Apply sigmoid to get values between 0 and 1
        heatmap = torch.sigmoid(heatmap)

        return heatmap, in_frame

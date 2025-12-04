import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# Backbone: ResNet18 + simple FPN (C3, C4, C5 -> P)
class BackboneFPN(nn.Module):
    def __init__(self, pretrained: bool = True, out_channels: int = 128):
        super().__init__()
        net = resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )
        self.layer1 = net.layer1   # C2
        self.layer2 = net.layer2   # C3
        self.layer3 = net.layer3   # C4
        self.layer4 = net.layer4   # C5

        self.lateral3 = nn.Conv2d(128, out_channels, 1)
        self.lateral4 = nn.Conv2d(256, out_channels, 1)
        self.lateral5 = nn.Conv2d(512, out_channels, 1)

        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: top-down, y: lateral
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral5(c5)
        p4 = self._upsample_add(p5, self.lateral4(c4))
        p3 = self._upsample_add(p4, self.lateral3(c3))

        p = self.out_conv(p3)  # (B, C, H', W')
        return p


# Straight (anchor-based) branch
class StraightNeck(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Placeholder: keep features as-is (can be replaced by convs if needed)
        self.proj = nn.Identity()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H', W')
        return self.proj(feat)


class StraightHead(nn.Module):
    """
    Predict lane-wise x coordinates per row + existence logits from spatial feature map.
    """

    def __init__(
        self,
        in_dim: int,
        num_lanes: int,
        num_rows: int,
        num_cols: int,
        num_cell_row: int,
        num_cell_col: int,
    ):
        super().__init__()
        self.num_lanes = num_lanes
        self.num_rows = num_rows

        in_channels = in_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # x regression head: (B, 64, H', W') -> (B, L, H', W')
        self.head_x = nn.Conv2d(64, num_lanes, kernel_size=1)

        # existence head: (B, 64, H', W') -> (B, L, H', W')
        self.head_exist = nn.Conv2d(64, num_lanes, kernel_size=1)

    def forward(self, feat: torch.Tensor, num_rows: int | None = None) -> dict:
        """
        Args:
            feat: (B, C, H', W')
        Returns:
            {
                "x": (B, L, R),          # x in pixels
                "exist_logit": (B, L, R)
            }
        """
        if num_rows is None:
            num_rows = self.num_rows

        x = self.conv(feat)  # (B, 64, Hf, Wf)

        x_reg = self.head_x(x)           # (B, L, Hf, Wf)
        exist_logit = self.head_exist(x) # (B, L, Hf, Wf)

        # Aggregate along vertical axis into R row anchors
        x_reg = F.adaptive_avg_pool2d(x_reg, (num_rows, 1)).squeeze(-1)        # (B, L, R)
        exist_logit = F.adaptive_avg_pool2d(exist_logit, (num_rows, 1)).squeeze(-1)  # (B, L, R)

        return {
            "x": x_reg,
            "exist_logit": exist_logit,
        }


# Bezier (curve) branch
class BezierNeck(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Global lane-aware feature vector
        x = self.block(feat)
        x = self.gap(x).flatten(1)  # (B, mid_channels)
        return x


class BezierHead(nn.Module):
    """
    Predict per-lane cubic Bezier control points + existence logits.
    """

    def __init__(self, in_dim: int, num_lanes: int):
        super().__init__()
        hidden = 256
        self.num_lanes = num_lanes

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.ctrl_out = nn.Linear(hidden, num_lanes * 4 * 2)
        self.exist_out = nn.Linear(hidden, num_lanes)

    def forward(self, gvec: torch.Tensor) -> dict:
        x = self.fc(gvec)
        ctrl = self.ctrl_out(x)  # (B, num_lanes * 4 * 2)
        ctrl = ctrl.view(-1, self.num_lanes, 4, 2)

        exist_logits = self.exist_out(x)  # (B, num_lanes)
        return {
            "ctrl_points": ctrl,
            "exist_logits": exist_logits,
        }


# Gating / routing head
class RoutingHead(nn.Module):
    """
    Predict per-lane, per-row gate g(b, l, r) in [0, 1] from backbone feature.
    """

    def __init__(self, in_channels: int, num_lanes: int, num_rows: int):
        super().__init__()
        self.num_lanes = num_lanes
        self.num_rows = num_rows

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_lanes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, num_rows: int | None = None) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H', W')
        Returns:
            gate: (B, L, R) in [0, 1]
        """
        if num_rows is None:
            num_rows = self.num_rows

        x = self.conv(feat)  # (B, L, H', W')
        x = F.adaptive_avg_pool2d(x, (num_rows, 1)).squeeze(-1)  # (B, L, R)
        gate = torch.sigmoid(x)
        return gate


# Full dual-head lane detection network
class DualHeadLaneNet(nn.Module):
    def __init__(
        self,
        num_lanes: int = 4,
        num_rows: int = 40,
        num_cols: int = 72,
        num_cell_row: int = 100,
        num_cell_col: int = 100,
        backbone_out_ch: int = 128,
    ):
        super().__init__()

        self.num_rows = num_rows

        self.backbone = BackboneFPN(pretrained=True, out_channels=backbone_out_ch)

        # Straight branch
        self.neck_straight = StraightNeck(in_channels=backbone_out_ch)
        self.head_straight = StraightHead(
            in_dim=backbone_out_ch,
            num_lanes=num_lanes,
            num_rows=num_rows,
            num_cols=num_cols,
            num_cell_row=num_cell_row,
            num_cell_col=num_cell_col,
        )

        # Bezier (curve) branch
        self.neck_curve = BezierNeck(in_channels=backbone_out_ch, mid_channels=128)
        self.head_curve = BezierHead(in_dim=128, num_lanes=num_lanes)

        # Routing / gating head
        self.routing_head = RoutingHead(
            in_channels=backbone_out_ch,
            num_lanes=num_lanes,
            num_rows=num_rows,
        )

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)  # (B, C, H', W')

        # Straight head
        feat_straight = self.neck_straight(feat)             # (B, C, H', W')
        out_straight = self.head_straight(feat_straight, self.num_rows)

        # Bezier head
        feat_curve = self.neck_curve(feat)                   # (B, mid)
        out_curve = self.head_curve(feat_curve)

        # Routing gate
        gate = self.routing_head(feat, self.num_rows)        # (B, L, R)

        return {
            "straight": out_straight,
            "bezier": out_curve,
            "gate": gate,
        }

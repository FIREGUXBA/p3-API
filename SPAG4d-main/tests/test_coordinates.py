# tests/test_coordinates.py
"""
Validate coordinate system transforms produce correct orientation
in target viewers.
"""

import numpy as np
import torch
import pytest

from spag4d.spherical_grid import create_spherical_grid, rotation_matrix_to_quaternion


class TestSphericalGrid:
    """Test spherical grid geometry."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    def test_equator_front(self, device):
        """
        Center pixel should point to +X in Y-up frame.
        
        At θ=π (center of image horizontally) and φ=π/2 (equator),
        the direction should be [sin(π/2)cos(π), cos(π/2), -sin(π/2)sin(π)]
        = [1*(-1), 0, -1*0] = [-1, 0, 0]
        
        Wait, let's recalculate. At image center (u=W/2, v=H/2):
        θ = (1 - (u + 0.5) / W) * 2π ≈ (1 - 0.5) * 2π = π
        φ = (v + 0.5) / H * π ≈ 0.5 * π = π/2
        
        r̂ = [sin(π/2)cos(π), cos(π/2), -sin(π/2)sin(π)]
           = [1*(-1), 0, -1*0] = [-1, 0, 0]
        
        So center points to -X, not +X. The front (θ=0) is at the right edge.
        """
        H, W = 180, 360
        grid = create_spherical_grid(H, W, device, stride=1)
        
        # Center of image (equator, θ=π)
        v_center, u_center = H // 2, W // 2
        rhat = grid.rhat[v_center, u_center].numpy()
        
        # Should point approximately to -X
        expected = np.array([-1., 0., 0.])
        assert np.allclose(rhat, expected, atol=0.05), f"Center r̂ = {rhat}, expected ≈ {expected}"
    
    def test_equator_right_edge(self, device):
        """
        Right edge of image (θ≈0) should point to +X.
        """
        H, W = 180, 360
        grid = create_spherical_grid(H, W, device, stride=1)
        
        # Right edge, equator
        v_center = H // 2
        u_right = W - 1
        rhat = grid.rhat[v_center, u_right].numpy()
        
        # θ = (1 - (W-0.5)/W) * 2π ≈ 0.5/W * 2π ≈ 0
        # r̂ ≈ [sin(π/2)cos(0), cos(π/2), -sin(π/2)sin(0)] = [1, 0, 0]
        expected = np.array([1., 0., 0.])
        assert np.allclose(rhat, expected, atol=0.1), f"Right edge r̂ = {rhat}, expected ≈ {expected}"
    
    def test_pole_direction(self, device):
        """Top row should point to +Y (up)."""
        H, W = 180, 360
        grid = create_spherical_grid(H, W, device, stride=1)
        
        # Top center
        rhat = grid.rhat[0, W // 2].numpy()
        
        # At φ ≈ 0: r̂ ≈ [0, 1, 0]
        assert rhat[1] > 0.95, f"Top row Y component = {rhat[1]}, expected > 0.95"
    
    def test_south_pole_direction(self, device):
        """Bottom row should point to -Y (down)."""
        H, W = 180, 360
        grid = create_spherical_grid(H, W, device, stride=1)
        
        # Bottom center
        rhat = grid.rhat[H - 1, W // 2].numpy()
        
        # At φ ≈ π: r̂ ≈ [0, -1, 0]
        assert rhat[1] < -0.95, f"Bottom row Y component = {rhat[1]}, expected < -0.95"
    
    def test_stride_reduces_dimensions(self, device):
        """Stride should reduce grid dimensions."""
        H, W = 360, 720
        
        grid_s1 = create_spherical_grid(H, W, device, stride=1)
        grid_s4 = create_spherical_grid(H, W, device, stride=4)
        
        assert grid_s1.theta.shape == (360, 720)
        assert grid_s4.theta.shape == (90, 180)
    
    def test_tangent_basis_orthonormal(self, device):
        """Tangent basis should be orthonormal."""
        H, W = 64, 128
        grid = create_spherical_grid(H, W, device, stride=1)
        
        # Sample a few points
        for v in [10, 32, 50]:
            for u in [10, 64, 100]:
                right = grid.tangent_right[v, u]
                up = grid.tangent_up[v, u]
                normal = -grid.rhat[v, u]
                
                # Check orthogonality
                assert torch.abs(torch.dot(right, up)) < 0.01
                assert torch.abs(torch.dot(right, normal)) < 0.01
                assert torch.abs(torch.dot(up, normal)) < 0.01
                
                # Check unit length
                assert torch.abs(right.norm() - 1) < 0.01
                assert torch.abs(up.norm() - 1) < 0.01
                assert torch.abs(normal.norm() - 1) < 0.01


class TestQuaternion:
    """Test quaternion utilities."""
    
    def test_identity_rotation(self):
        """Identity matrix should give identity quaternion."""
        R = torch.eye(3).unsqueeze(0)
        q = rotation_matrix_to_quaternion(R)
        
        # Identity quaternion: [0, 0, 0, 1] in XYZW
        expected = torch.tensor([[0., 0., 0., 1.]])
        assert torch.allclose(q.abs(), expected.abs(), atol=0.01)
    
    def test_90_deg_rotation_about_z(self):
        """90° rotation about Z should give correct quaternion."""
        # R_z(90°)
        angle = torch.tensor(np.pi / 2)
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=torch.float32).unsqueeze(0)
        
        q = rotation_matrix_to_quaternion(R)
        
        # Expected: sin(45°)*[0,0,1], cos(45°) in XYZW
        # = [0, 0, 0.707, 0.707]
        expected_z = np.sin(np.pi / 4)
        expected_w = np.cos(np.pi / 4)
        
        assert torch.abs(q[0, 2]) - expected_z < 0.01  # Z component
        assert torch.abs(q[0, 3]) - expected_w < 0.01  # W component
    
    def test_batch_conversion(self):
        """Batch conversion should work."""
        batch_size = 10
        R = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
        q = rotation_matrix_to_quaternion(R)
        
        assert q.shape == (batch_size, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import os
import torch
import numpy as np
from PHASE import PHASE
import matplotlib.pyplot as plt

class RSP:
    """
    RSP class handles the near-field to far-field transformation for a given lens system.

    Attributes:
        lens (PHASE): The lens object containing phase and amplitude information.
        cx (float): X-coordinate of the center point.
        cy (float): Y-coordinate of the center point.
        screen_size (float): Size of the screen in micrometers. Must be a positive value.
        sampling (int): Number of sampling points. Must be a positive value.
        device (str): Device to use for computations ('cpu' or 'gpu').
        X (ndarray): X-coordinates for the sampling points, shape(sampling,).
        Y (ndarray): Y-coordinates for the sampling points, shape(sampling,).
        UZ (ndarray): Complex field after transformation, initially set to None.
        ita (float): Not currently used.
        Z (ndarray): Z-coordinates for the sampling points, initially set to None.
        RSP_lambda (float): Wavelength for the RSP calculations, initially set to None.
        
    Raises:
        ValueError: If screen_size or sampling are not positive values.
    
    Methods:
        RSP_XY_GPU: Performs the near-field to far-field transformation on the XY plane using GPU.
        draw_XY: Plots the transformed field in the XY plane.
        RSP_XZ_GPU: Performs the near-field to far-field transformation on the XZ plane using GPU.
        draw_XZ: Plots the transformed field in the XZ plane.
    """
    def __init__(self, 
                 lens = PHASE, 
                 cx = 0, 
                 cy = 0, 
                 screen_size = 30, 
                 sampling = 201, 
                 device='cpu'):
        
        if screen_size <= 0 or sampling <= 0:
            raise ValueError("screen_size and sampling must be positive values.")
        
        self.device = device
        self.lens = lens
        self.phase_type = lens.phase_type
        self.cx = cx
        self.cy = cy
        self.screen_size = screen_size
        self.sampling = sampling

        self.RSP_lambda = None
        self.X, self.Y = np.meshgrid(np.linspace(-self.screen_size / 2, self.screen_size / 2, self.sampling), np.linspace(-self.screen_size / 2, self.screen_size / 2, self.sampling))
        self.UXY = None
        
        self.ita = None
        self.Z = None
        self.UXZ = None
        

    def RSP_XY_GPU(self, RSP_lambda = 0.532, z = 100.):
        """
        Perform the near-field to far-field transformation on the XY plane using GPU.

        Args:
            RSP_lambda (float): Wavelength for the RSP calculations.
            z (float): Distance for the transformation.

        Returns:
            None
        """
        self.RSP_lambda = RSP_lambda
        self.Z = z

        try:
            k = 2 * np.pi / self.RSP_lambda
            U_temp = torch.zeros(1, len(self.Y), dtype=torch.complex64, device=self.device)
            U = torch.zeros(len(self.X), len(self.Y), dtype=torch.complex64, device=self.device)
            LENS_X = torch.tensor(self.lens.X, dtype=torch.float32, device=self.device)
            LENS_Y = torch.tensor(self.lens.Y, dtype=torch.float32, device=self.device)
            GPUZ = torch.tensor(self.Z, dtype=torch.float32, device=self.device)
            GPUX = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            GPUY = torch.tensor(self.Y, dtype=torch.float32, device=self.device)
            GPUU0 = torch.tensor(self.lens.U0, dtype=torch.complex64, device=self.device)
            w,h = GPUX.shape
            for a in range(len(self.X)):
                for b in range(len(self.Y)):
                    r01 = torch.sqrt(GPUZ**2 + (GPUX[a, b] - LENS_X)**2 + (GPUY[a, b] - LENS_Y)**2)
                    U_temp[0, b] = torch.sum(GPUZ / (1j * self.RSP_lambda) * GPUU0 * torch.exp(1j * k * r01) / r01**2)
                U[a, :] = U_temp

            self.UXY = U.cpu().numpy()
        except Exception as e:
            print(f"Error during GPU computation: {e}")
            raise

    def draw_XY(self):
        """
        Plot the transformed field in the XY plane.

        This method creates a contour plot of the transformed field's intensity in the XY plane.

        Attributes:
            contour (QuadContourSet): The contour plot of the phase profile.
            cbar (Colorbar): The colorbar associated with the contour plot.

        Returns:
            None
        """
        plt.figure("Farfield_XY", figsize=(9, 7.2))
        contour = plt.pcolormesh(self.X, self.Y, np.abs(self.UXY)**2, vmin = 0, cmap='CMRmap', shading='gouraud')
        cbar = plt.colorbar(contour)
        #cbar.set_ticks(np.linspace(0, 2*np.pi, num=7))
        #cbar.set_ticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        cbar.ax.tick_params(labelsize=24, labelfontfamily='Times New Roman')
        plt.xlabel('x (µm)', fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.ylabel('y (µm)', fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.title('', fontsize=26, fontweight='bold', fontname='Times New Roman')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=24)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
        plt.axis('equal')
        plt.axis('tight')
        plt.text(self.cx - 0.5 * (np.max(self.X) - np.min(self.X)),
                 self.cy + 0.4 * (np.max(self.Y) - np.min(self.Y)),
                 f"λ = {int(self.RSP_lambda*1000)} nm",fontsize=20,rotation=0,bbox={
        'boxstyle':'round',
        'facecolor':'#fff',
        'edgecolor':None,
        'pad':0.5,
        'linewidth':1})
        plt.savefig(f"./result/"+self.phase_type+f"/xy-plane/lambda={int(self.RSP_lambda*1000)},z={self.Z}.png",
            transparent=True,
            bbox_inches='tight',
            pad_inches=1)
        plt.show()
        np.savez(f"./result/"+self.phase_type+f"/xy-plane/lambda={int(self.RSP_lambda*1000)},z={self.Z}.npz", X = self.X, Y = self.Y, IXY = np.abs(self.UXY)**2)

    def RSP_XZ_GPU(self, RSP_lambda = 0.532, focal_Z = np.arange(10, 151, 1)):
        """
        Perform the near-field to far-field transformation on the XZ plane using GPU.

        Args:
            RSP_lambda (float): Wavelength for the RSP calculations.
            z (float): Distance for the transformation.

        Returns:
            None
        """
        self.RSP_lambda = RSP_lambda
        self.Z = focal_Z

        try:
            k = 2 * np.pi / self.RSP_lambda
            U_temp = torch.zeros(1, len(self.X), dtype=torch.complex64, device=self.device)
            U = torch.zeros(len(self.Z), len(self.X), dtype=torch.complex64, device=self.device)

            LENS_X = torch.tensor(self.lens.X, dtype=torch.float32, device=self.device)
            LENS_Y = torch.tensor(self.lens.Y, dtype=torch.float32, device=self.device)
            GPUKersi = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            GPUIta = torch.tensor(self.Y, dtype=torch.float32, device=self.device)
            GPUU0 = torch.tensor(self.lens.U0, dtype=torch.complex64, device=self.device)
            for c in range(len(self.Z)):
                z_val = torch.tensor(self.Z[c], dtype=torch.float32, device=self.device)

                for a in range(len(self.X)):
                    b = len(GPUIta) // 2
                    r01 = torch.sqrt(z_val**2 + (GPUKersi[b, a] - LENS_X)**2 + (GPUIta[b, a] - LENS_Y)**2)
                    U_temp[0, a] = torch.sum(z_val / (1j * self.RSP_lambda) * GPUU0 * torch.exp(1j * k * r01) / r01**2)
                    U[c, :] = U_temp

            self.UXZ = U.cpu().numpy()
        except Exception as e:
            print(f"Error during GPU computation: {e}")
            raise

    def draw_XZ(self):
        """
        Plot the transformed field in the XY plane.

        This method creates a contour plot of the transformed field's intensity in the XY plane.

        Attributes:
            contour (QuadContourSet): The contour plot of the phase profile.
            cbar (Colorbar): The colorbar associated with the contour plot.

        Returns:
            None
        """
        plt.figure("Farfield_XY", figsize=(18, 6))
        
        Z, X = np.meshgrid(self.Z, np.linspace(-self.screen_size / 2, self.screen_size / 2, self.sampling))
        contour = plt.pcolormesh(Z, X, np.transpose(np.abs(self.UXZ)**2), vmin=0, cmap='CMRmap', shading='gouraud')
        cbar = plt.colorbar(contour)
        #cbar.set_ticks(np.linspace(0, 2*np.pi, num=7))
        #cbar.set_ticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        cbar.ax.tick_params(labelsize=24, labelfontfamily='Times New Roman')
        plt.xlabel('z (µm)', fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.ylabel('x (µm)', fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.title('', fontsize=26, fontweight='bold', fontname='Times New Roman')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.tick_params(axis='both', which='major', labelsize=24)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
        ax.set_aspect('equal', 'box')
        plt.text(min(self.Z) + 0.05 * (np.max(self.Z) - np.min(self.Z)),
                 self.cy + 0.4 * (np.max(X) - np.min(X)),
                 f"λ = {int(self.RSP_lambda*1000)} nm",fontsize=20,rotation=0,bbox={
        'boxstyle':'round',
        'facecolor':'#fff',
        'edgecolor':None,
        'pad':0.5,
        'linewidth':1})
        plt.savefig(f"./result/"+self.phase_type+f"/xz-plane/lambda={int(self.RSP_lambda*1000)}.png",
            transparent=True,
            bbox_inches='tight',
            pad_inches=1)
        plt.show()
        np.savez (f"./result/"+self.phase_type+f"/xz-plane/lambda={int(self.RSP_lambda*1000)}.npz", Z = Z, X = X, IXZ = np.transpose(np.abs(self.UXZ)**2))
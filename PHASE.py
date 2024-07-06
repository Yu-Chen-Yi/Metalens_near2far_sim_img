import numpy as np
import matplotlib.pyplot as plt

class PHASE:
    """
    PHASE class represents a phase profile for a lens system.

    Args:
    lens_diameter (float): The diameter of the lens in micrometers.
    focal_length (float): The focal length of the lens in micrometers.
    design_lambda (float): The design wavelength in micrometers.
    alpha (float): The coefficient for the cubic phase term.
    phase_type (str): The type of phase profile to use. Can be 'EP', 'cubic', or 'abs_cubic'.
    device (str): The device to use for calculations, default is 'cpu'.

    Raises:
        ValueError: If lens_diameter, focal_length, or design_lambda are not positive values.

    Methods:
        calculate_phase: Calculates the phase profile based on the initialized attributes.
        draw: Plots the phase profile using matplotlib.
    """
    def __init__(self, lens_diameter = 50., 
                 focal_length = 100., 
                 design_lambda=0.532, 
                 alpha=0, 
                 phase_type='EP', 
                 device='cpu'):
        # Raises
        if lens_diameter <= 0 or focal_length <= 0 or design_lambda <= 0:
            raise ValueError("lens_diameter, focal_length, and design_lambda must be positive values.")
        
        self.device = device
        self.lens_diameter = lens_diameter
        self.focal_length = focal_length
        self.design_lambda = design_lambda
        self.alpha = alpha
        self.phase_type = phase_type
        self.x = np.arange(-self.lens_diameter / 2, self.lens_diameter / 2, 0.05)
        self.y = self.x
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.amplitude = np.zeros(self.X.shape)
        self.phase = np.zeros(self.X.shape)
        self.U0 = np.zeros(self.X.shape)
        self.calculate_phase()

    def calculate_phase(self):
        """
        Calculate the phase profile based on the lens parameters.

        This function computes the phase profile using the given lens diameter, focal length,
        design wavelength, and other parameters. The phase profile is stored in the attribute `self.phase`.
        
        Attributes:
            aperture (ndarray, shape(self.X.shape)): Binary mask representing the lens aperture.
            phase (ndarray, shape(self.X.shape)): Calculated phase profile of the lens.
            E (ndarray, shape(self.X.shape)): Complex field with amplitude and phase information.
            amplitude (ndarray, shape(self.X.shape)): Amplitude of the complex field.
            U0 (ndarray, shape(self.X.shape)): Initial complex field with phase and amplitude.

        Phase Types:
            'cubic': Phase profile includes cubic terms.
            'abs_cubic': Phase profile includes absolute cubic terms.
            'EP': Phase profile is calculated based on the lens equation.
            
        Returns:
            None
        """

        aperture = (self.X**2 + self.Y**2) < (self.lens_diameter / 2)**2
        if self.phase_type == 'cubic':
            self.phase = -2 * np.pi / self.design_lambda * (
                np.sqrt(self.X**2 + self.Y**2 + self.focal_length**2) - self.focal_length +
                self.alpha / (self.lens_diameter / 2)**3 * (self.X**3 + self.Y**3)
            )
        if self.phase_type == 'abs_cubic':
            self.phase = -2 * np.pi / self.design_lambda * (
            np.sqrt(self.X**2 + self.Y**2 + self.focal_length**2) - self.focal_length +
            self.alpha / (self.lens_diameter / 2)**3 * (np.abs(self.X**3) + np.abs(self.Y**3))
            )
        if self.phase_type == 'EP':
            self.phase = -2 * np.pi / self.design_lambda * (
            np.sqrt(self.X**2 + self.Y**2 + self.focal_length**2) - self.focal_length
            )
        E = 1 * np.exp(1j * self.phase) * aperture
        self.phase = np.mod(np.angle(E), 2 * np.pi) * aperture
        self.amplitude = np.abs(E) * aperture
        self.U0 = self.amplitude * np.exp(1j * self.phase)

    def draw(self):
        """
        Plot the phase profile using matplotlib.

        This method creates a 2D plot of the phase profile stored in the attribute `self.phase`.
        The plot displays the phase distribution across the lens surface.

        Attributes:
            contour (QuadContourSet): The contour plot of the phase profile.
            cbar (Colorbar): The colorbar associated with the contour plot.

        Returns:
            None
        """
        plt.figure("Phase", figsize=(9, 7.2))
        contour = plt.pcolormesh(self.X, self.Y, self.phase, vmin=self.phase.min(), vmax=self.phase.max(), cmap='CMRmap', shading='gouraud')
        cbar = plt.colorbar(contour)
        cbar.set_ticks(np.linspace(0, 2*np.pi, num=7))
        cbar.set_ticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
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
        plt.savefig(f"./result/"+self.phase_type+f"/Phase_design_lambda={int(self.design_lambda*1000)}_self.lens_diameter={self.lens_diameter}_focal_length={self.focal_length}.png",
            transparent=True,
            bbox_inches='tight',
            pad_inches=1)
        plt.show()
        np.savez(f"./result/"+self.phase_type+f"/Phase_design_lambda={int(self.design_lambda*1000)}_self.lens_diameter={self.lens_diameter}_focal_length={self.focal_length}.npz", X = self.X, Y = self.Y, phase = self.phase)
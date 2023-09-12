import numpy as np
from hyperspy.component import Component

class Gaussian2D(Component):

    """
    """

    def __init__(self, A=1, cx=0, cy=0, sx=1, sy=1, angle=0):
        # Define the parameters
        Component.__init__(self, ('A', 'cx', 'cy', 'sx','sy','angle'))

        # Optionally we can set the initial values
        self.A.value = A
        self.cx.value = cx
        self.cy.value = cy
        self.sx.value = sx
        self.sy.value = sy
        self.angle.value = np.deg2rad(angle)


    # Define the function as a function of the already defined parameters,
    # x being the independent variable value
    def function(self, x, y):
        A = self.A.value
        cx = self.cx.value
        cy = self.cy.value
        sx = self.sx.value
        sy = self.sy.value
        u = self.angle.value
        
        a = (np.cos(u)/sx)**2/2 + (np.sin(u)/sy)**2/2
        b = -np.sin(2*u)/(4*sx**2) + np.sin(2*u)/(4*sy**2)
        c = (np.sin(u)/sx)**2/2 + (np.cos(u)/sy)**2/2
        return A*np.exp(-a*(x-cx)**2 - 2*b*(x-cx)*(y-cy) - c*(y-cy)**2)
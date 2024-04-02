"""Ivmech PID Controller is simple implementation of a Proportional-Integral-Derivative (PID) Controller in the Python Programming Language.
More information about PID Controller: http://en.wikipedia.org/wiki/PID_controller
"""

class PID:
    """PID Controller
    """
    def __init__(self):
        self.setpoint = 0.5
        self.delta_time = 0.001
        self.action_bound = [0.6, 1]  # [P_bound, I_bound]
        self.SF_lower = 0.01   # stability factor
        self.clear()

    def clear(self, P=0.008, I=0.3):
        """Clears PID computations and coefficients"""
        self.Kp = P
        self.Ki = I
        self.dP = 0.0
        self.dI = 0.0
        self.last_error = 0.0
        self.ErrorSum = 0.0
        self.output = 0.0

    def update(self, deflV, last_deflV, last_error_sum):
        """Calculates PID value for given reference feedback"""
        self.last_error = self.setpoint - last_deflV
        error           = self.setpoint - deflV
        self.PTerm      = self.Kp * error
        error_sum       = self.last_error * self.delta_time * 0.5 + last_error_sum
        self.ITerm      = 0.5 * self.delta_time * error + error_sum
        self.output     = self.PTerm + (self.Ki * self.ITerm)
        return error_sum

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

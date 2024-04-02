from PID import PID


class Env(PID):
    
    def __init__(self):
        super().__init__()
        self.lower_Volt = -1
        self.upper_Volt = 15
        self.DeflV_bound = 0.5
        self.ZVolt_bound = 0.1

        
    def step(self, action, last_state, DeflV, last_error_sum):
        """Calculates PI value"""
        self.dP = action[0]
        self.dI = action[1]
        self.Kp *= (1 + self.dP)
        self.Ki *= (1 + self.dI)
        print('self.Kp:', self.Kp)
        print('self.Ki:', self.Ki)
        print("last_state in step:", last_state)
        last_DeflV = last_state[0]
        last_ZVolt = last_state[1]
        error_sum = self.update(DeflV, last_DeflV, last_error_sum)
        ZVolt = self.output + last_ZVolt
        # Deflection Voltage exceeds voltage bounds (delta_V in one time step)
        if DeflV > self.setpoint * (1+self.DeflV_bound):
            reward = -10
        # PID output exceeds voltage bounds (delta_PIDop in one time step)
        elif abs(self.output) > self.ZVolt_bound*(self.upper_Volt-self.lower_Volt):
            reward = -4
        # Z Piezo exceeds maximum deformation
        elif ZVolt >= self.upper_Volt:
            ZVolt = self.upper_Volt
            reward = -4
        # Z Piezo exceeds minimum deformation
        elif ZVolt <= self.lower_Volt:
            ZVolt = self.lower_Volt
            reward = -4
        # check if error is within a range
        else:
            reward = 5 if abs(self.last_error) < 0.2 else -2
        state = [round(DeflV, 3), round(ZVolt, 3), self.Kp, self.Ki]
        return state, reward, error_sum


############
# Visit data struct collect all data from a single patient visit
#################

class VisitData:
    def __init__(self):
        # 4 byte array
        self.udprs = np.empty((0))   
        self.visit_number = np.empty((0))
        # prptides content per visit empty dictionary
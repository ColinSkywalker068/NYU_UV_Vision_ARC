class distance_mapper:
    def __init__(self, distance_array) -> None:
        self.front = distance_array[0]
        self.left = distance_array[1]
        self.right = distance_array[2]
        self.back = distance_array[3]

    def get_front_distance(self):
        return self.front
    def get_left_distance(self):
        return self.left
    def get_right_distance(self):
        return self.right 
    def get_all_distances(self):
        return [self.front, self.left, self.right, self.back]
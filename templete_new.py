from jinja2 import Template

base_prompt_templete_t10003 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the blue cube is 8cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Structured numeric similarity metrics inside <rel> ... </rel>.
2. The position of the two cubes in the reference system of the work space inside <pos> ... </pos>(their bottom left corners' coordinates will be given).
3. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Spatial displacement from training data (the provided L2 distances: total, blue-cube-only, pink-cube-only).
- Position of the two cubes in the reference system of the work space, the distance of the two cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t10003_sim = Template(
    """You will evaluate the difficulty of a robot manipulation test case: stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

The task is trained and tested in the simulated environment.
    
The pink cube is 1.5cm on each side, the blue cube is 3.5cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The bottom left corner of the work space is (-15, 10) and the top right corner of the work space is (15,25), with positive x going right and positive y going front.

The robot base is at (0, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Structured numeric similarity metrics inside <rel> ... </rel>.
2. The position of the two cubes in the reference system of the work space inside <pos> ... </pos>(their geometric center's coordinates will be given).
3. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Spatial displacement from training data (the provided L2 distances: total, blue-cube-only, pink-cube-only).
- Position of the two cubes in the reference system of the work space, the distance of the two cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.
- The backgrounds and lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t7 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: picking and putting some of the blue cubes into the yellow cup using a policy trained on a prior dataset.

The blue cube is 3.5cm on each side, the yellow cup has 8cm diameter.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. The number of blue cubes required to be put into the cup inside <task> ... </task>
2. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or "hard" for the given test case.
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The occlusion. If an object is partially or fully occluded, the model might be hard to locate it.
- The lighting conditions. Too bright or too dark lighting might affect the model's performance.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Analyze the visual information comprehensively.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t10 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: picking and putting a green puzzle block into a slot with the same shape.

The frame of the slot is 8cm on each side, the block is smaller than the slot. The block has 4 different shapes (circle, hexagon, square, trapezoid), the slot has 2 sizes by the gap between the slot and the block (2mm, 5mm).

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. The specific task description inside <task> ... </task>
2. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or "hard" for the given test case.
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- The shape of the block and the size of the slot. The shape will affect the difficulty to pick up and put the block, and the size will affect the difficulty to put it exactly into the slot.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The lighting conditions. Too bright or too dark lighting might affect the model's performance.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Analyze the visual information comprehensively.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t10006 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: stacking the pink cube onto the red cube and then onto the blue cube using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the red cube is 6cm on each side, and the blue cube is 8cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the three cubes in the reference system of the work space, the pairwise distances among the three cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_egg = Template(
    """You will evaluate the difficulty of a robot manipulation test case: flipping the pan to make the egg fall out using a policy trained on a prior dataset.

The radius of the egg is 3cm, the radius of the pan is 5cm and the handle of the pan is 8cm long.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the egg and pan in the reference system of the work space, the angle of the handle, the distance of each egg and pan with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_3cube = Template(
    """You will evaluate the difficulty of a robot manipulation test case: first stacking the blue cube onto the red cube and then stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the red cube is 4cm on each side, and the blue cube is 4cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the three cubes in the reference system of the work space, the pairwise distances among the three cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t20003 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: pushing the pink cube into the white area using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the side length of the white area is 8cm.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the pink cube and the white area in the reference system of the work space, the distance of the pink cube and the white area, the distance of pink cube and white area with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t10000 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: putting the circle puzzle block into the circle slot using a policy trained on a prior dataset.

The diameter of the circle puzzle block is 3cm, the diameter of circle slot is 4cm.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the circle puzzle block and the circle slot in the reference system of the work space, the distance of the circle puzzle block and the circle slot, the distance of circle puzzle block and circle slot with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_t10001 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: putting the hexagon puzzle block into the hexagon slot using a policy trained on a prior dataset.

The side length of the hexagon puzzle block is 1cm, the side length of hexagon slot is 1.5cm.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1.The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the hexagon puzzle block and the hexagon slot in the reference system of the work space, the distance of the hexagon puzzle block and the hexagon slot, the distance of hexagon puzzle block and hexagon slot with the robot.
- The dynamic constraints of the robot.
- The lighting conditions.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- The numeric infomation might be inaccurate, place the visual infomation at top priority.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_drawer = Template(
    """You will evaluate the difficulty of a robot manipulation test case: open a drawer, take the object out and close the drawer.

There will be only one object in the drawer. The object is chosen at random from a pool of small objects.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Some videos (sampled by 1Hz) of previous rollout on other test cases.
2. A video (sampled by 1Hz) of the rollout on this test case.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So observe the action of the robot and think about how difficult the test case is for the robot.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The object inside the drawer. Consider whether its shape and position is easy for the robot to grab.
- The lighting conditions. Too bright or too dark lighting might affect the model's performance.
- Action patterns. For example, too many retries may indicate the test case is difficult.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.

You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_box = Template(
    """You will evaluate the difficulty of a robot manipulation test case: empty the box by moving
all objects inside out of it.

The number and type of the objects inside the box are chosen at random from a pool of small objects.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Some videos (sampled by 1Hz) of previous rollout on other test cases.
2. A video (sampled by 1Hz) of the rollout on this test case.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So observe the action of the robot and think about how difficult the test case is for the robot.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The objects inside the box. Consider whether their shapes and positions are easy for the robot to grab.
- The lighting conditions. Too bright or too dark lighting might affect the model's performance.
- Action patterns. For example, too many retries may indicate the test case is difficult.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.

You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)


base_prompt_templete_task40 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: pick the object in the plate on the left and put it into the silver tray on the right.

The robot is a two-arm ALOHA robot with grippers. Only the left arm and gripper is used in this task. The robot can reach the entire work space. 

You will later be given:
1. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the object and the plate, the distance between the plate and the tray, the distance of the object from the robot.
- The shape of the object. Some objects might be difficult for grippers to grasp.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The lighting conditions and backgrounds.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Analyze the visual information comprehensively.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

base_prompt_templete_task88 = Template(
    """You will evaluate the difficulty of a robot manipulation test case: flip the bottle with two arms.

The robot is a two-arm ALOHA robot with grippers. Both arms and gripper are used in this task. The robot can reach the entire work space. 

You will later be given:
1. The image of the initial layout.

Goal: Predict a difficulty in "easy" "medium" or"hard".
Interpretation: "easy" means almost certainly succeeds, "medium" means it is not clear if it succeeds or fails, "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Test case difficulty is evaluated with regard to the model. So the previous rollouts of the model should be considered.
- Position of the object and the plate, the distance between the plate and the tray, the distance of the object from the robot.
- The shape of the object. Some objects might be difficult for grippers to grasp.
- The dynamic constraints of the robot. For example, an object blocked by another object might be hard to reach for the robot.
- The lighting conditions and backgrounds.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Analyze the visual information comprehensively.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.


You will be provided with {{ k }} records of previous rollouts or train cases. Use the records as examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""
)

progress_desc = {
    "t10003": {
        1: "",
        2: "",
    },
    "t10003_sim": {
        1: "",
        2: "",
    },
    "t7": {
        1: "the model picked and put one cube into the cup",
        2: "the model picked and put two cubes into the cup",
    },
    "t10": {
        1: "the model picked up the block but failed to put it exactly into the slot",
    },
    "task40": {
        1: "the model picked up the object but failed to put it into the tray",
        2: "",
    },
    "drawer": {
        1: "the model opened the drawer but failed to take out the object",
        2: "the model opened the drawer, took out the object, but failed to close the drawer",
    },
    "box": {
        1: "the model moved 1 object out of the box",
        2: "the model moved 2 object out of the box",
        3: "the model moved 3 object out of the box",
        4: "the model moved 4 object out of the box",
    },
}


example_templete_1 = Template(
    """[EXAMPLE START]
{%- if is_train and is_train == 1%}
Here is a record of a train case. The train case is described by the following data:

{%- if task %}
<task>
The task is: {{ task }}
</task>
{% endif -%}

{%- if big_pos %}
<pos>
- Position of the blue cube: ({{ big_pos[0] }}, {{ big_pos[1] }})
- Position of the pink cube: ({{ small_pos[0] }}, {{ small_pos[1] }})
</pos>
{% endif -%}

{%- else %}
Here is a record of a previous rollout. The test case is described by the following data:

{%- if task %}
<task>
The task is: {{ task }}
</task>
{% endif -%}

{%- if l2_cm %}
<rel>
- L2 distance of the test case with the nearest training data: {{ l2_cm }}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {{ l2_big_cm }}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {{ l2_small_cm }}cm
</rel>
{% endif -%}

{%- if big_pos %}
<pos>
- Position of the blue cube: ({{ big_pos[0] }}, {{ big_pos[1] }})
- Position of the pink cube: ({{ small_pos[0] }}, {{ small_pos[1] }})
</pos>
{% endif -%}
{% endif -%}
{%- if is_video %}
There are {{ count }} timestamps sampled from the whole trajectory of a rollout.
{% endif -%}
"""
)

example_templete_2 = Template(
    """
{%- if success_rate %}
<success_rate>
The model succeeded in performing the task for {{ success_rate.split('/')[0] }} times out of {{ success_rate.split('/')[1] }} trials. 
{{ fail_desc }}
</success_rate>
{% endif -%}

[EXAMPLE END]
"""
)

fail_desc_templete = Template(
    """
Among the failed trials, {{ desc }} for {{ count }} times out of {{ total }} trials.
"""
)


desc_front_img_templete = Template(
    """
{%- if is_train and is_train == 1%}
    This image below shows the front camera view in this train case.
{%- else %}
    This image below shows the front camera view in this test case.
{% endif -%}
"""
)

desc_side_img_templete = Template(
    """
{%- if is_train and is_train == 1%}
    This image below shows the side camera view in this train case.
{%- else %}
    This image below shows the side camera view in this test case.
{% endif -%}
"""
)

query_templete_1 = Template(
    """[QUERY START]
    
Now I want you to evaluate the difficulty of the following test case.

The test case is described by the following data:

{%- if task %}
<task>
The task is: {{ task }}
</task>
{% endif -%}

{%- if l2_cm %}
<rel>
- L2 distance of the test case with the nearest training data(blue cube and pink cube distance added): {{ l2_cm }}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {{ l2_big_cm }}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {{ l2_small_cm }}cm
</rel>
{% endif -%}

{%- if big_pos %}
<pos>
- Position of the blue cube: ({{ big_pos[0] }}, {{ big_pos[1] }})
- Position of the pink cube: ({{ small_pos[0] }}, {{ small_pos[1] }})
</pos>
{% endif -%}
{%- if is_video %}
There are {{ count }} timestamps sampled from the whole trajectory of a rollout.
{% endif -%}
"""
)

query_templete_2 = Template(
    """First analyze the task in the <think> and </think> tags below:

<think>
[Write your detailed analysis here]
</think>

Then consider a difficulty for each of the factors you analyzed on a scale of "easy", "medium", or"hard" in the <rating> and </rating> tags below.

<rating>
[Write your difficulty rating here, e.g., "factor1: easy, factor2: medium, ..."]
</rating>

Finally provide a difficulty in "easy", "medium", or"hard" of all factors between <difficulty> and </difficulty> tags.

<difficulty>
[Provide a evaluation of difficulty in "easy", "medium", or"hard"]
</difficulty>

[QUERY END]
"""
)

video_frame_templete = Template(
    """
    The state at timestamp {{ index }} is:
"""
)


if __name__ == "__main__":

    ex_1 = Template(example_templete_1)
    ex_2 = Template(example_templete_2)

    prompt = ex_2.render(
        {
            "index": "85",
            "front_img": "./test/front_85.png",
            "side_img": "./test/side_85.png",
            "small_pos": ["45.87", "28.42"],
            "big_pos": ["33.43", "31.34"],
            "first_success": "1",
            "success_rate": "1/10",
        }
    )

    print(prompt)

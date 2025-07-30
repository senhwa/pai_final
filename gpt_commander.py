#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyexpat.errors import messages
import rospy
import os
import json
import numpy as np
from triton_main.msg import AuvStatus, AuvCommand
from openai import OpenAI
import time

import requests
import subprocess
import os

class TritonTTS:
    def __init__(self):
        self.tts_url = "http://localhost:50021"

    def generate_audio(self, text, speaker=1):
        query = requests.post(
            f"{self.tts_url}/audio_query",
            params={"text": text, "speaker": speaker},
        ).json()

        wav_response = requests.post(
            f"{self.tts_url}/synthesis",
            params={"speaker": speaker},
            json=query
        )

        if wav_response.status_code == 200:
            subprocess.run(["aplay", "-"], input=wav_response.content, text=False)
        else:
            raise Exception(f"Failed to generate audio")

class AUVVelocityController:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        rospy.init_node('gpt_auv_velocity_controller', anonymous=True)

        self.status_sub = rospy.Subscriber('/status/auv_status', AuvStatus, self.status_callback)
        self.cmd_pub = rospy.Publisher('/control/command_auv', AuvCommand, queue_size=10)

        self.current_status = None

        self.target_x = 153.0
        self.target_y = -50.0
        self.target_depth = 3.0  # Target depth in meters
        
        self.tts = TritonTTS()
        self.tts_flag = 1
        self.last_response_text = ""

        self.timer = rospy.Timer(rospy.Duration(1.5), self.timer_callback)
        rospy.sleep(1.0)  # Allow time for subscribers to connect
        self.timer = rospy.Timer(rospy.Duration(4.0), self.timer_callback_with_tts)

    def status_callback(self, msg: AuvStatus):
        self.current_status = msg
        
    def timer_callback(self, event):
        if not self.current_status:
            rospy.loginfo("Waiting for AUV status...")
            return

        
        manually_calculated_distance = np.hypot(self.target_x - self.current_status.xyz_ins[0],
                                                self.target_y - self.current_status.xyz_ins[1])
        relative_angle_deg = np.degrees(np.arctan2(self.target_y - self.current_status.xyz_ins[1],
                                                    self.target_x - self.current_status.xyz_ins[0])) - self.current_status.heading
        relative_angle_deg = (relative_angle_deg + 180) % 360 - 180  # Normalize to [-180, 180]
        
        user_prompt = (
            f"Current: x={self.current_status.xyz_ins[0]:.2f}, y={self.current_status.xyz_ins[1]:.2f}, z={self.current_status.xyz_ins[2]:.2f}, yaw_deg={self.current_status.heading:.2f}\n"
            f"Target: x={self.target_x:.2f}, y={self.target_y:.2f}, z={self.target_depth:.2f}\n"
        )
        
        user_prompt += f"\ndistance: {manually_calculated_distance:.2f} m, relative angle: {relative_angle_deg:.2f} deg"
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You control an AUV. output local-frame velocity: vx in [0.0, 0.5] m/s, yaw_rate in [-5.0, 5.0] deg/s"
                    "set vx_mps to move closer to the target, never set zero before reaching the target with in 1.0 m"
                    "If the relative angle is positive, MUST set yaw_rate to positive, and if the relative angle is negative, MUST set yaw_rate to negative."
                    "If the abs relative angle larger than 30 degrees, MUST set vx zero and turn first."
                    "You strongly keep this rule, never output any other value for target yaw."
                    "Try to move the AUV towards the target and stop it at the target position"
                    "You can set depth command for auv to dive"
                    "If the distance is larger than 50 m, must keep the desired depth 0.0 m to prevent collision and max surge velocity"
                    "If the distance is less than 50 m, must set to desired depth to the depth of the target wrecked ship."
                    "If the distance is within 2m, set zero vx_mps, keeping yaw and depth."
                    
                    "Stop the AUV at the relative distance with in 1.0m and zero velocity and zero yaw rate. "
                    "Output JSON in this format:\n"
                    "{"
                    "\"distance\": ..., "
                    "\"relative_angle_deg\": ..., "
                    "\"vx_mps\": ..., "
                    "\"yaw_rate\": ..., "
                    "\"desired_depth\": ..., "
                    "}\n"
                    "only output JSON, do not output any other text"
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            called_time = rospy.get_time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.0,
                max_tokens=100,
            )

            reply = response.choices[0].message.content.strip()
            rospy.loginfo(f"GPT output: {reply}")
            elapsed_time = rospy.get_time() - called_time
            rospy.loginfo(f"GPT call took {elapsed_time:.2f} seconds")

            result = json.loads(reply)

            command = AuvCommand()
            command.header.stamp = rospy.Time.now()
            command.xyz_ref.mode = [2,0,3]
            command.xyz_ref.ref = [float(result["vx_mps"]), 0.0, float(result["desired_depth"])]
            command.rpy_ref.mode = [0, 0, 2]
            command.rpy_ref.ref = [0.0, 0.0, float(result["yaw_rate"])]
            self.cmd_pub.publish(command)
        except Exception as e:
            rospy.logwarn(f"GPT call or parse failed: {e}")

    def timer_callback_with_tts(self, event):
        if not self.current_status:
            rospy.loginfo("Waiting for AUV status...")
            return

        
        manually_calculated_distance = np.hypot(self.target_x - self.current_status.xyz_ins[0],
                                                self.target_y - self.current_status.xyz_ins[1])
        relative_angle_deg = np.degrees(np.arctan2(self.target_y - self.current_status.xyz_ins[1],
                                                    self.target_x - self.current_status.xyz_ins[0])) - self.current_status.heading
        relative_angle_deg = (relative_angle_deg + 180) % 360 - 180  # Normalize to [-180, 180]
        
        user_prompt = (
            f"Current: x={self.current_status.xyz_ins[0]:.2f}, y={self.current_status.xyz_ins[1]:.2f}, z={self.current_status.xyz_ins[2]:.2f}, yaw_deg={self.current_status.heading:.2f}\n"
            f"Target: x={self.target_x:.2f}, y={self.target_y:.2f}, z={self.target_depth:.2f}\n"
        )

        
        user_prompt += f"\ndistance: {manually_calculated_distance:.2f} m, relative angle: {relative_angle_deg:.2f} deg"
        messages = [
            {
                "role": "system",
                "content": (
                    "you are a very inteligent and clever AUV professional and gentle captain, add one sentence for your status breifing, in Japanese with must in 50 characters"
                    "your mission is finding wrecked ship 沈没船 and stop at the target position with in 1.0 m"
                    "express everything in japanese, never use english even alphabet"
                    "use words like 深度, ヨー角, 距離, 相対角度 and never mention units"
                    "Mainly metion one of the auv status (velocity, depth, target yaw, distance, relative angle) in your speech, randomly pick only one of them, do not metion all of them"
                    "Sometimes you can mention the target wrecked ship, but do not mention the target position"
                    "add a funny comment for mission"
                    "say unit in hiragana, like めーとる"
                    "Respond ONLY with the JSON object, do not add any extra text before or after."
                    "If the relative distance is smaller than 2m you can say that you are very happy and report mission success and found the target wrecked ship"
                    "must output JSON in this format:\n"
                    "{"
                    "\"briefing\": ..., "
                    "}"
                    "and do not output any other text"
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            called_time = rospy.get_time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.0,
                max_tokens=100,
            )

            reply = response.choices[0].message.content.strip()
            json_reply = json.loads(reply)
            print(f"GPT output: {reply}")

            if self.tts_flag == 1:
                try:
                    self.tts.generate_audio(json_reply["briefing"], speaker=1)
                except Exception as e:
                    rospy.logwarn(f"TTS generation failed: {e}")

        except Exception as e:
            rospy.logwarn(f"GPT call or parse failed: {e}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    controller = AUVVelocityController()
    controller.run()

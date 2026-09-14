#!/usr/bin/env python3
"""Preview or update only an existing Pi orchestrator's served capabilities.

Use a locally saved, verified serving profile after final qualification. Shares
the recipe installer's lock, private backups, atomic writes and restore format.
Does not install a fleet, change defaults, or alter worker definitions.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path

import install


def plan(raw, profile):
    if profile.get("id")!="glm-5.3-flash-exl3-ablit" or "dgx-orchestrator" not in profile.get("aliases",[]):
        raise ValueError("profile is not the fleet orchestrator")
    context=profile["model"]["context_tokens"]
    modalities=profile["model"]["input_modalities"]
    if context not in (500000,750000,850000) or modalities!=["text","image"]:
        raise ValueError("profile is outside the image qualification matrix")
    models=json.loads(raw)
    matches=[m for m in models["providers"]["dgx-spark"]["models"] if m["id"]=="dgx-orchestrator"]
    if len(matches)!=1: raise ValueError("expected exactly one existing orchestrator route")
    model=matches[0]
    before={k:copy.deepcopy(model.get(k)) for k in ("input","contextWindow")}
    model.update(input=modalities,contextWindow=context)
    return install.encoded(models),{"before":before,"after":{k:model[k] for k in before}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir",type=Path,default=Path.home()/".pi/agent")
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--expected-sha256",help="models.json digest from reviewed preview; required to apply")
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    if args.apply and not args.expected_sha256: parser.error("apply requires the reviewed models.json digest")
    home=args.agent_dir.expanduser().resolve()
    profile_raw=args.profile.read_bytes(); profile=json.loads(profile_raw)
    def prepare():
        path=install.target(home,"models.json"); raw=path.read_bytes()
        if args.expected_sha256 and install.sha(raw)!=args.expected_sha256:
            raise ValueError("models.json changed since preview")
        content,preview=plan(raw,profile)
        result={"models_sha256":install.sha(raw),"profile_sha256":install.sha(profile_raw),**preview}
        if args.apply:
            if path.read_bytes()!=raw: raise ValueError("concurrent models.json change")
            result.update(install.apply(home,{"models.json":content}))
        else: result["preview"]=True
        return result
    if args.apply:
        with (home/".spark-serve-recipe.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=prepare()
    else: result=prepare()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()

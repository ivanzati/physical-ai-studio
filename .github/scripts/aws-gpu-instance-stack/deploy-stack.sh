#!/usr/bin/env bash
#
# Deploys cf-template-ec2-gpu.yaml into the VPC/subnet created by create-vpc.sh.
# Reads VpcId and SubnetId from the ids file that create-vpc.sh wrote.
#
# Usage:
#   ./deploy-stack.sh up   --stack gpu-01 [--ids-file gpu-net.ids.env]
#                          [--region us-east-2] [--profile geti-prod]
#                          [--instance-type g4dn.xlarge]
#   ./deploy-stack.sh down --stack gpu-01 [--region us-east-2] [--profile geti-prod]
#
# 'up'   creates the stack, waits for completion and prints the outputs.
# 'down' deletes the stack (does NOT touch the VPC; use create-vpc.sh down for that).
#
# Run multiple machines by calling 'up' with a different --stack name each time.

set -euo pipefail

REGION="${AWS_REGION:-us-east-2}"
PROFILE="${AWS_PROFILE:-}"
STACK=""
IDS_FILE="gpu-net.ids.env"
INSTANCE_TYPE=""
TEMPLATE="cf-template-ec2-gpu.yaml"

die() { echo "Error: $*" >&2; exit 1; }
need() { [[ -n "${2-}" ]] || die "option $1 requires a value"; }

aws_() {
  local args=(--region "$REGION" --no-cli-pager)
  [[ -n "$PROFILE" ]] && args+=(--profile "$PROFILE")
  # Global options must precede the command/args or the CLI may misparse them.
  aws "${args[@]}" "$@"
}

cmd="${1:-}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack)         need "$1" "${2-}"; STACK="$2"; shift 2 ;;
    --ids-file)      need "$1" "${2-}"; IDS_FILE="$2"; shift 2 ;;
    --region)        need "$1" "${2-}"; REGION="$2"; shift 2 ;;
    --profile)       need "$1" "${2-}"; PROFILE="$2"; shift 2 ;;
    --instance-type) need "$1" "${2-}"; INSTANCE_TYPE="$2"; shift 2 ;;
    --template)      need "$1" "${2-}"; TEMPLATE="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$STACK" ]] || die "--stack <name> is required"

deploy() {
  [[ -f "$TEMPLATE" ]]  || die "template not found: $TEMPLATE (run from the repo directory)"
  [[ -f "$IDS_FILE" ]]  || die "ids file not found: $IDS_FILE (run create-vpc.sh up first)"
  # shellcheck disable=SC1090
  source "$IDS_FILE"
  [[ -n "${VPC_ID:-}"    ]] || die "VPC_ID missing from $IDS_FILE"
  [[ -n "${SUBNET_ID:-}" ]] || die "SUBNET_ID missing from $IDS_FILE"

  local params=(ParameterKey=VpcId,ParameterValue="$VPC_ID"
                ParameterKey=SubnetId,ParameterValue="$SUBNET_ID")
  [[ -n "$INSTANCE_TYPE" ]] && params+=(ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE")

  echo "Creating stack '$STACK' in $REGION (VpcId=$VPC_ID, SubnetId=$SUBNET_ID)..."
  # CAPABILITY_NAMED_IAM is accepted even when the template has no IAM resources,
  # so the script keeps working if IAM roles/instance profiles are added later.
  aws_ cloudformation create-stack \
    --stack-name "$STACK" \
    --template-body "file://$TEMPLATE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --tags Key=purpose,Value=gpu-instance \
    --parameters "${params[@]}" \
    --query 'StackId' --output text

  echo "Waiting for stack to complete (drivers are preinstalled; smoke test only)..."
  if ! aws_ cloudformation wait stack-create-complete --stack-name "$STACK"; then
    echo "Stack did not complete. Recent failure events:" >&2
    aws_ cloudformation describe-stack-events --stack-name "$STACK" \
      --query "StackEvents[?contains(ResourceStatus,'FAILED')].[LogicalResourceId,ResourceStatusReason]" \
      --output text >&2
    exit 1
  fi

  echo "Stack '$STACK' ready. Outputs:"
  aws_ cloudformation describe-stacks --stack-name "$STACK" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output text
}

destroy() {
  echo "Deleting stack '$STACK' in $REGION..."
  aws_ cloudformation delete-stack --stack-name "$STACK"
  aws_ cloudformation wait stack-delete-complete --stack-name "$STACK"
  echo "Stack '$STACK' deleted. (VPC left intact; use create-vpc.sh down to remove it.)"
}

case "$cmd" in
  up)   deploy ;;
  down) destroy ;;
  *)    die "usage: $0 {up|down} --stack <name> [options]  (see header for details)" ;;
esac

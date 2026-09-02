#!/usr/bin/env bash
#
# Creates the minimal VPC networking (VPC + Internet Gateway + public subnet +
# route table) required by cf-template-ec2-gpu.yaml and prints the VpcId and
# SubnetId to pass as CloudFormation parameters.
#
# Usage:
#   ./create-vpc.sh up     [--region us-east-2] [--profile geti-prod] [--name gpu-net] [--ids-file gpu-net.ids.env]
#   ./create-vpc.sh down    --ids-file gpu-net.ids.env
#
# 'up'   creates the resources, writes their IDs to the --ids-file path (default
#        <name>.ids.env), and prints the parameter values plus a ready-to-run
#        create-stack command.
# 'down' deletes everything recorded in the given ids file.

set -euo pipefail

REGION="${AWS_REGION:-us-east-2}"
PROFILE="${AWS_PROFILE:-}"
NAME="gpu-net"
VPC_CIDR="10.42.0.0/16"
SUBNET_CIDR="10.42.1.0/24"
IDS_FILE=""

die() { echo "Error: $*" >&2; exit 1; }
need() { [[ -n "${2-}" ]] || die "option $1 requires a value"; }

aws_() {
  local args=(--region "$REGION" --output text --no-cli-pager)
  [[ -n "$PROFILE" ]] && args+=(--profile "$PROFILE")
  # Global options must precede the command/args or the CLI may misparse them.
  aws "${args[@]}" "$@"
}

cmd="${1:-}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)   need "$1" "${2-}"; REGION="$2"; shift 2 ;;
    --profile)  need "$1" "${2-}"; PROFILE="$2"; shift 2 ;;
    --name)     need "$1" "${2-}"; NAME="$2"; shift 2 ;;
    --vpc-cidr) need "$1" "${2-}"; VPC_CIDR="$2"; shift 2 ;;
    --subnet-cidr) need "$1" "${2-}"; SUBNET_CIDR="$2"; shift 2 ;;
    --ids-file) need "$1" "${2-}"; IDS_FILE="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

create_network() {
  local ids_file="${IDS_FILE:-${NAME}.ids.env}"
  : > "$ids_file"

  local az
  az=$(aws_ ec2 describe-availability-zones --query 'AvailabilityZones[0].ZoneName')

  local vpc_id
  vpc_id=$(aws_ ec2 create-vpc --cidr-block "$VPC_CIDR" \
    --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'Vpc.VpcId')
  echo "VPC_ID=$vpc_id" >> "$ids_file"
  aws_ ec2 modify-vpc-attribute --vpc-id "$vpc_id" --enable-dns-support >/dev/null
  aws_ ec2 modify-vpc-attribute --vpc-id "$vpc_id" --enable-dns-hostnames >/dev/null

  local igw_id
  igw_id=$(aws_ ec2 create-internet-gateway \
    --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'InternetGateway.InternetGatewayId')
  echo "IGW_ID=$igw_id" >> "$ids_file"
  aws_ ec2 attach-internet-gateway --internet-gateway-id "$igw_id" --vpc-id "$vpc_id"

  local subnet_id
  subnet_id=$(aws_ ec2 create-subnet --vpc-id "$vpc_id" --cidr-block "$SUBNET_CIDR" \
    --availability-zone "$az" \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'Subnet.SubnetId')
  echo "SUBNET_ID=$subnet_id" >> "$ids_file"
  aws_ ec2 modify-subnet-attribute --subnet-id "$subnet_id" --map-public-ip-on-launch >/dev/null

  local rtb_id
  rtb_id=$(aws_ ec2 create-route-table --vpc-id "$vpc_id" \
    --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'RouteTable.RouteTableId')
  echo "RTB_ID=$rtb_id" >> "$ids_file"
  aws_ ec2 create-route --route-table-id "$rtb_id" \
    --destination-cidr-block 0.0.0.0/0 --gateway-id "$igw_id" >/dev/null
  local assoc_id
  assoc_id=$(aws_ ec2 associate-route-table --route-table-id "$rtb_id" \
    --subnet-id "$subnet_id" --query 'AssociationId')
  echo "ASSOC_ID=$assoc_id" >> "$ids_file"

  # Single-line optional flag avoids fragile multi-line ${PROFILE:+...} expansion.
  local profile_flag=""
  [[ -n "$PROFILE" ]] && profile_flag="--profile $PROFILE "

  cat <<EOF

Networking ready (region: $REGION, AZ: $az). IDs saved to: $ids_file

CloudFormation parameter values:
  VpcId    = $vpc_id
  SubnetId = $subnet_id

Deploy the stack:
  aws cloudformation create-stack \\
    --stack-name gpu-instance \\
    --template-body file://cf-template-ec2-gpu.yaml \\
    --region $REGION ${profile_flag}\\
    --parameters ParameterKey=VpcId,ParameterValue=$vpc_id \\
                 ParameterKey=SubnetId,ParameterValue=$subnet_id

Tear down networking later:
  ./create-vpc.sh down --ids-file $ids_file --region $REGION ${profile_flag}
EOF
}

delete_network() {
  [[ -n "$IDS_FILE" ]] || die "down requires --ids-file <file>"
  [[ -f "$IDS_FILE" ]] || die "ids file not found: $IDS_FILE"
  # shellcheck disable=SC1090
  source "$IDS_FILE"

  [[ -n "${ASSOC_ID:-}"  ]] && aws_ ec2 disassociate-route-table --association-id "$ASSOC_ID" || true
  [[ -n "${RTB_ID:-}"    ]] && aws_ ec2 delete-route-table --route-table-id "$RTB_ID" || true
  [[ -n "${SUBNET_ID:-}" ]] && aws_ ec2 delete-subnet --subnet-id "$SUBNET_ID" || true
  if [[ -n "${IGW_ID:-}" && -n "${VPC_ID:-}" ]]; then
    aws_ ec2 detach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID" || true
    aws_ ec2 delete-internet-gateway --internet-gateway-id "$IGW_ID" || true
  fi
  [[ -n "${VPC_ID:-}" ]] && aws_ ec2 delete-vpc --vpc-id "$VPC_ID" || true

  rm -f "$IDS_FILE"
  echo "Networking deleted."
}

case "$cmd" in
  up)   create_network ;;
  down) delete_network ;;
  *)    die "usage: $0 {up|down} [options]  (see header for details)" ;;
esac

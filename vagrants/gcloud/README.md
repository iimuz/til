# gcloud in vagrant

vm installed gcloud using vagrant

## install GCloud

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## setup GCloud

```bash
$ gcloud init
$ gcloud auth application-default login
```

## Usage terraform

```bash
$ terraform init
$ terraform plan
$ terraform apply
```

## SSH

```bash
$ gcloud compute ssh <instance name>
```

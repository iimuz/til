# static web site using firebase

firebase を利用した静的サイトの構築デモ。

## 手順

### 初期化

`firebase init` コマンドを実行するといくつかの質問に答えることで、プロジェクトの初期化ができる。

```sh
$ firebase init

     ######## #### ########  ######## ########     ###     ######  ########
     ##        ##  ##     ## ##       ##     ##  ##   ##  ##       ##
     ######    ##  ########  ######   ########  #########  ######  ######
     ##        ##  ##    ##  ##       ##     ## ##     ##       ## ##
     ##       #### ##     ## ######## ########  ##     ##  ######  ########

You're about to initialize a Firebase project in this directory:

  /src

Before we get started, keep in mind:

  * You are currently outside your home directory

? Which Firebase CLI features do you want to set up for this folder? Press Space to select features, then
Enter to confirm your choices.
 ◯ Database: Deploy Firebase Realtime Database Rules
 ◯ Firestore: Deploy rules and create indexes for Firestore
 ◯ Functions: Configure and deploy Cloud Functions
❯◉ Hosting: Configure and deploy Firebase Hosting sites
 ◯ Storage: Deploy Cloud Storage security rules

=== Project Setup

First, let's associate this project directory with a Firebase project.
You can create multiple project aliases by running firebase use --add,
but for now we'll just set up a default project.

? Please select an option:
  Use an existing project
  Create a new project
  Add Firebase to an existing Google Cloud Platform project
❯ Don't set up a default project

=== Hosting Setup

Your public directory is the folder (relative to your project directory) that
will contain Hosting assets to be uploaded with firebase deploy. If you
have a build process for your assets, use your build's output directory.

? What do you want to use as your public directory? (public)
have a build process for your assets, use your build's output directory.

? What do you want to use as your public directory? public
? Configure as a single-page app (rewrite all urls to /index.html)? No
✔  Wrote public/404.html
✔  Wrote public/index.html

i  Writing configuration info to firebase.json...
i  Writing project information to .firebaserc...
i  Writing gitignore file to .gitignore...

✔  Firebase initialization complete!
```

## プロジェクトの追加

上記ではデフォルトプロジェクトを追加しなかったので、下記コマンドで手動で追加する。

```sh
$ firebase use --add

? Which project do you want to add? hoge
? What alias do you want to use for this project? (e.g. staging) staging

Created alias staging for hoge.
Now using alias staging (hoge)
```

設定は、 `.firebase` に追加される。

## プロジェクトの変更

プロジェクトを変更したい場合は、下記のように実行する。

```sh
$ firebase use staging
```

## その他

- デプロイ: `firebase deploy`
- 公開停止: `firebase hosting:disable`


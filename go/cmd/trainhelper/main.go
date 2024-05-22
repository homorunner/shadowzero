package main

import (
	"context"
	"ddt/pkg/config"
	"ddt/pkg/db"
	"ddt/pkg/repo"
	"ddt/pkg/repo/dataset"
	"ddt/pkg/repo/gating"
	"ddt/pkg/repo/model"
	"flag"
	"log"
	"os"
	"path"
	"time"
)

var cfgFile string
var module string
var action string
var arguments []string

func init() {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}

	flag.StringVar(&cfgFile, "c", path.Join(homeDir, ".ddt", ".config.toml"), "config file path")
	flag.Parse()

	module = flag.Arg(0)
	action = flag.Arg(1)
	arguments = flag.Args()[2:]

	time.Local = time.UTC
}

func main() {
	cfg, err := config.LoadFromFile(cfgFile)
	if err != nil {
		log.Fatal("load config failed", err)
	}

	ctx := context.Background()

	log.Print("connecting to db")
	db, err := db.Connect(ctx, &cfg.DBConfig)
	if err != nil {
		log.Fatal("connect to db failed", err)
	}
	defer db.Close(ctx)
	log.Print("connected to db")

	repoArgs := repo.RepoArgs{
		DB: db,
	}

	switch module {
	case "model":
		modelRepo := model.New(repoArgs)
		modelRepo.Handle(ctx, action, arguments)
	case "gating":
		gatingRepo := gating.New(repoArgs)
		gatingRepo.Handle(ctx, action, arguments)
	case "dataset":
		datasetRepo := dataset.New(repoArgs)
		datasetRepo.Handle(ctx, action, arguments)
	default:
		log.Fatal("invalid module", module)
	}
}

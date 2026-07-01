from tunnel_cloud.cli import main

if __name__ == "__main__":
    main(["features", *(__import__("sys").argv[1:])])

// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
// and what to do when importing types
declare namespace App {
  interface Locals {}

  // interface PageData {}
  // interface Error {}

  // Include support for environment variables.
  // The `env` object, containing KV namespaces and other storage objects,
  // is passed to SvelteKit via the platform property along with context
  // and caches, meaning you can access it in hooks and endpoints.
  interface Platform {
    env: {
      COUNTER: DurableObjectNamespace;
    };
    context: {
      waitUntil(promise: Promise<any>): void;
    };
    caches: CacheStorage & { default: Cache };
  }

  interface Session {}

  interface Stuff {}
}
